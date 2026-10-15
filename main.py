
# =============================================================================
# Temporal Convolutional Network (TCN)
# =============================================================================
#
# ## 1. TCNとは？
#
# TCN（Temporal Convolutional Network）は、時系列データの時間依存関係を
# 1次元畳み込み（Conv1d）によって学習するニューラルネットワークです。
#
# RNNやLSTMのように時刻を順番に処理するのではなく、
# 畳み込みを利用して複数時刻の情報を並列に処理します。
#
# ### TCNの基本的な構成要素
#
# 1. Causal Convolution（因果畳み込み）
#    - 時刻tの出力に、未来の時刻t+1以降の情報を使用しない。
#
# 2. Dilated Convolution（拡張畳み込み）
#    - 時間方向に間隔を空けて畳み込みを実行する。
#    - 少ない層数でも長期間の時間依存関係を捉えられる。
#
# 3. Residual Connection（残差接続）
#    - 畳み込み処理前の入力を出力に加算する。
#    - 深いネットワークでも勾配を伝播しやすくする。
#
# 4. Weight Normalization（重み正規化）
#    - 重みの大きさと方向を分離してパラメータ化する。
#    - 学習を安定させる目的で使用される。
#
#
# ## 2. TCN全体のデータフロー
#
# 入力時系列 X
#     │
#     ▼
# TemporalBlock 1 (dilation=1)
#     │
#     ▼
# TemporalBlock 2 (dilation=2)
#     │
#     ▼
# TemporalBlock 3 (dilation=4)
#     │
#     ▼
# TemporalBlock 4 (dilation=8)
#     │
#     ▼
# 時系列特徴表現 H
#
# ※ TemporalBlockの数はnum_channelsの要素数で決まります。
#
# 入力テンソル：
#     X.shape = (Batch, Features, Time)
#
# 出力テンソル：
#     H.shape = (Batch, OutputChannels, Time)
#
# 例：
#     入力  : (32, 6, 100)
#     出力  : (32, 64, 100)
#
# Batch          = 32サンプル
# Features       = 6種類の入力特徴量
# Time           = 100時刻
# OutputChannels = 64種類の学習された特徴表現
#
# 注意：
# Conv1dでは (Batch, Channels, Time) の順番を使用します。
# 一般的なテーブル形式 (Batch, Time, Features) とは異なります。
#
# =============================================================================


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# =============================================================================
# ## 3. Chomp1d：未来情報の混入を防ぐための切り取り処理
# =============================================================================
#
# ### 3.1 なぜChomp1dが必要なのか？
#
# PyTorchのConv1dでは、paddingを指定すると
# 時系列の左右にゼロが追加されます。
#
# 例えば：
#
# 元の時系列：
#     [x1, x2, x3, x4, x5]
#
# padding=2の場合：
#     [0, 0, x1, x2, x3, x4, x5, 0, 0]
#
# このまま通常の畳み込みを実行すると、
# 時刻tの出力が未来の入力に依存する可能性があります。
#
# TCNでは未来情報を使わない因果性を保証するため、
# 畳み込み後の右端を切り取ります。
#
#
# ### 3.2 Causal Convolutionの考え方
#
# 通常の畳み込み：
#
#     y[t] = f(x[t-1], x[t], x[t+1])
#
# → 未来のx[t+1]を使用する可能性がある。
#
# 因果畳み込み：
#
#     y[t] = f(x[t-2], x[t-1], x[t])
#
# → 現在と過去の情報だけを使用する。
#
#
# ### 3.3 Chomp1dの役割
#
# この実装では、
#
#     Conv1d(padding=p)
#             ↓
#     Chomp1d(chomp_size=p)
#
# という組み合わせによって因果畳み込みを実現しています。
#
# kernel_size=k, dilation=d, stride=1の場合：
#
#     p = (k - 1) * d
#
# 畳み込み後の長さは：
#
#     L_conv = L_input + p
#
# Chomp1dで右端p個を除去すると：
#
#     L_output = L_input
#
# つまり、
# 1. 未来情報を使用しない
# 2. 入力と出力の時系列長を一致させる
#
# という2つの条件を満たします。
#
# ※ この説明は本実装のstride=1を前提としています。
# ※ chomp_size=0では[:-0]が空になるため、0への対応はありません。
#
# =============================================================================


class Chomp1d(nn.Module):
    """
    Conv1dの出力から右端の要素を取り除くモジュール。

    Parameters
    ----------
    chomp_size : int
        時間方向の右端から除去する要素数。

    Input
    -----
    x : Tensor
        shape = (Batch, Channels, Time)

    Output
    ------
    Tensor
        shape = (Batch, Channels, Time - chomp_size)
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()

        # 時間方向の末尾から削除する要素数
        self.chomp_size = chomp_size

    def forward(self, x):

        # x[:, :, :-self.chomp_size]
        #
        # 第1軸 : Batch → 全サンプルを維持
        # 第2軸 : Channel → 全チャネルを維持
        # 第3軸 : Time → 末尾chomp_size個を削除
        #
        # contiguous()はメモリ上で連続したテンソルを返します。
        return x[:, :, :-self.chomp_size].contiguous()


# =============================================================================
# ## 4. TemporalBlock：TCNの基本構成単位
# =============================================================================
#
# TemporalBlockはTCNの中心となるモジュールです。
#
# 1つのTemporalBlockには、以下の処理が含まれます。
#
# 1. Dilated Causal Convolution
# 2. Weight Normalization
# 3. ReLU
# 4. Dropout
# 5. 2回目のDilated Causal Convolution
# 6. Residual Connection
#
#
# ### 4.1 TemporalBlockの内部構造
#
#                         Input X
#                            │
#              ┌─────────────┴─────────────┐
#              │                           │
#              ▼                           │
#          Conv1d                          │
#              │                           │
#           Chomp1d                        │
#              │                           │
#             ReLU                         │
#              │                           │
#           Dropout                        │
#              │                           │
#              ▼                           │
#          Conv1d                          │
#              │                           │
#           Chomp1d                        │
#              │                           │
#             ReLU                         │
#              │                           │
#           Dropout                        │
#              │                           │
#              ▼                           ▼
#              │                    Identity / Conv1d
#              │                           │
#              └───────────┬───────────────┘
#                          │
#                        Add
#                          │
#                         ReLU
#                          │
#                          ▼
#                        Output
#
#
# ### 4.2 Dilated Convolutionとは？
#
# 通常のConv1dでは、連続する時刻の値を使用します。
#
# kernel_size=3, dilation=1：
#
#     [x(t-2), x(t-1), x(t)]
#
# dilation=2：
#
#     [x(t-4), x(t-2), x(t)]
#
# dilation=4：
#
#     [x(t-8), x(t-4), x(t)]
#
# dilationを大きくすると、パラメータ数を大幅に増やさずに
# より遠い過去の情報を参照できます。
#
# 一般式：
#
#     y[t] = Σ(k=0 to K-1) w[k] * x[t - k*d]
#
# K : kernel_size
# d : dilation
# w : 畳み込みフィルタの重み
#
# 実際には入力チャネルと出力チャネルについても和を取ります。
#
#
# ### 4.3 Residual Connectionとは？
#
# 通常のネットワーク：
#
#     y = F(x)
#
# 残差接続：
#
#     y = ReLU(F(x) + x)
#
# この実装では、入力と出力のチャネル数が異なる場合、
# 1×1畳み込みによってチャネル数を揃えます。
#
#     y = ReLU(F(x) + W_res(x))
#
# 主な目的：
# - 深いネットワークでの勾配伝播を改善する
# - 入力情報を後段に伝えやすくする
# - 残差成分を学習できるようにする
#
# =============================================================================


class TemporalBlock(nn.Module):
    """
    TCNの基本構成単位となるResidual Block。

    Parameters
    ----------
    n_inputs : int
        入力チャネル数。

    n_outputs : int
        出力チャネル数。

    kernel_size : int
        畳み込みフィルタのサイズ。

    stride : int
        畳み込みの移動幅。
        このTCNでは通常1を指定する。

    dilation : int
        畳み込み時に参照する時刻の間隔。

    padding : int
        時間方向に追加するゼロの数。

    dropout : float
        Dropoutの確率。

    Input
    -----
    (Batch, n_inputs, Time)

    Output
    ------
    (Batch, n_outputs, Time)

    ※ stride=1かつ適切なpaddingを指定した場合。
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2
    ):

        super(TemporalBlock, self).__init__()

        # ---------------------------------------------------------------------
        # 4.4 第1畳み込み層
        # ---------------------------------------------------------------------
        #
        # 入力：
        #     (Batch, n_inputs, Time)
        #
        # 出力：
        #     (Batch, n_outputs, Time + padding)
        #
        # ※ stride=1、padding=(kernel_size-1)*dilationの場合。
        #
        # weight_norm：
        #     重みを大きさと方向に分離してパラメータ化します。
        #
        # dilation：
        #     時間方向の参照間隔を指定します。
        #
        # padding：
        #     Chomp1dと組み合わせて因果性を実現します。
        #
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )

        # Conv1dで生じた右端の余分な時刻を削除
        self.chomp1 = Chomp1d(padding)

        # 非線形性を導入
        self.relu1 = nn.ReLU()

        # 過学習を抑制
        self.dropout1 = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # 4.5 第2畳み込み層
        # ---------------------------------------------------------------------
        #
        # 第1畳み込み層と同じdilationを使用します。
        #
        # 2つの畳み込み層を重ねることで、
        # 1つのTemporalBlockが参照できる過去の範囲を広げます。
        #
        # kernel_size=2, dilation=4の場合：
        #
        # 1層目：
        #     現在と4時刻前を参照
        #
        # 2層目：
        #     1層目の出力をさらに4時刻ずらして参照
        #
        # 結果：
        #     最大8時刻前までの情報が出力に影響する。
        #
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # 4.6 畳み込み処理をSequentialにまとめる
        # ---------------------------------------------------------------------
        #
        # 処理順序：
        #
        # Conv1d
        #   ↓
        # Chomp1d
        #   ↓
        # ReLU
        #   ↓
        # Dropout
        #   ↓
        # Conv1d
        #   ↓
        # Chomp1d
        #   ↓
        # ReLU
        #   ↓
        # Dropout
        #
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2
        )

        # ---------------------------------------------------------------------
        # 4.7 Residual Connection用のチャネル変換
        # ---------------------------------------------------------------------
        #
        # 入力と出力のチャネル数が一致する場合：
        #
        #     Residual = x
        #
        # 一致しない場合：
        #
        #     Residual = Conv1d(x, kernel_size=1)
        #
        # 1×1畳み込みは時間方向の近傍を混合せず、
        # 各時刻のチャネル方向の線形変換を行います。
        #
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )

        # 残差加算後に適用する活性化関数
        self.relu = nn.ReLU()

        # 重み初期化
        self.init_weights()

    # -------------------------------------------------------------------------
    # 4.8 重み初期化
    # -------------------------------------------------------------------------
    #
    # 正規分布 N(0, 0.01^2) に基づいて重みを初期化します。
    #
    # 平均：
    #     μ = 0
    #
    # 標準偏差：
    #     σ = 0.01
    #
    # 注意：
    # この実装はweight_norm適用後のweight.dataを操作しています。
    # weight_normは重みを再パラメータ化するため、
    # この代入が内部の学習パラメータを意図どおり初期化するとは
    # 限りません。
    #
    # また、バイアスはここでは明示的に初期化していません。
    # =============================================================================

    def init_weights(self):

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    # -------------------------------------------------------------------------
    # 4.9 順伝播
    # -------------------------------------------------------------------------
    #
    # 数式：
    #
    #     F(x) = Dropout(
    #                ReLU(
    #                  CausalConv2(
    #                    Dropout(
    #                      ReLU(CausalConv1(x))
    #                    )
    #                  )
    #                )
    #            )
    #
    #     Residual(x) = x または Conv1d_1x1(x)
    #
    #     Output = ReLU(F(x) + Residual(x))
    #
    # -------------------------------------------------------------------------

    def forward(self, x):

        # 2層のDilated Causal Convolutionを通す
        out = self.net(x)

        # 入力チャネル数と出力チャネル数を揃える
        res = (
            x
            if self.downsample is None
            else self.downsample(x)
        )

        # 残差加算後にReLUを適用
        return self.relu(out + res)


# =============================================================================
# ## 5. TemporalConvNet：TemporalBlockを積み重ねるネットワーク
# =============================================================================
#
# TemporalConvNetは複数のTemporalBlockを順番に接続します。
#
# 最大の特徴は、各ブロックでdilationを指数的に増加させることです。
#
#
# ### 5.1 dilationの設定
#
#     dilation_size = 2 ** i
#
# i=0 → dilation=1
# i=1 → dilation=2
# i=2 → dilation=4
# i=3 → dilation=8
# i=4 → dilation=16
#
# このように指数的に拡張することで、
# 少ない層数で長期間の時間依存関係を扱えます。
#
#
# ### 5.2 Receptive Field（受容野）
#
# Receptive Fieldとは、
# ある時刻の出力に影響を与えることができる
# 入力時系列の時間範囲を指します。
#
# この実装では1つのTemporalBlockに
# 同じdilationを持つConv1dが2層あります。
#
# stride=1、kernel_size=K、
# 各ブロックのdilationが2^iの場合：
#
#     RF = 1 + 2 * (K - 1) * Σ(i=0 to L-1) 2^i
#
# 等比数列の和より：
#
#     RF = 1 + 2 * (K - 1) * (2^L - 1)
#
# L = TemporalBlockの数
# K = kernel_size
#
# 例えばkernel_size=2の場合：
#
#     L=1 → RF=3
#     L=2 → RF=7
#     L=3 → RF=15
#     L=4 → RF=31
#     L=5 → RF=63
#     L=6 → RF=127
#
# 6ブロックで最大127時刻分の情報を参照できます。
#
# ただし、実際の入力時系列が127時刻より短い場合、
# 参照範囲の一部はゼロパディングになります。
#
#
# ### 5.3 num_channelsの意味
#
# num_channelsは各TemporalBlockの出力チャネル数を指定します。
#
# 例：
#
#     num_inputs = 6
#     num_channels = [16, 32, 64]
#
# ネットワーク構造：
#
#     Input (6 channels)
#         ↓
#     TemporalBlock 1
#       6 → 16 channels
#       dilation = 1
#         ↓
#     TemporalBlock 2
#       16 → 32 channels
#       dilation = 2
#         ↓
#     TemporalBlock 3
#       32 → 64 channels
#       dilation = 4
#         ↓
#     Output (64 channels)
#
# ここでのチャネル数は、元の特徴量数とは異なります。
#
# 入力特徴量が6種類であっても、
# 畳み込みによって16、32、64種類の潜在特徴表現を
# 学習することができます。
#
# また、全ての入力チャネルが畳み込みによって混合されます。
# そのため、このTCNは特徴量ごとに独立した
# 時間依存性だけを学習する構造ではありません。
#
# 時間方向のパターンと特徴量間の相互作用を
# 同時に学習できる構造です。
#
# =============================================================================


class TemporalConvNet(nn.Module):
    """
    複数のTemporalBlockを積み重ねたTCN本体。

    Parameters
    ----------
    num_inputs : int
        入力特徴量数。

    num_channels : list[int]
        各TemporalBlockの出力チャネル数。

    kernel_size : int
        畳み込みカーネルサイズ。
        デフォルトは2。

    dropout : float
        Dropout率。
        デフォルトは0.2。

    Input
    -----
    x.shape = (Batch, num_inputs, Time)

    Output
    ------
    y.shape = (Batch, num_channels[-1], Time)
    """

    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=2,
        dropout=0.2
    ):

        super(TemporalConvNet, self).__init__()

        # TemporalBlockを格納するリスト
        layers = []

        # TemporalBlockの数
        num_levels = len(num_channels)

        # ---------------------------------------------------------------------
        # 5.4 TemporalBlockを順番に構築
        # ---------------------------------------------------------------------

        for i in range(num_levels):

            # dilationを指数的に増加
            #
            # i=0 → 1
            # i=1 → 2
            # i=2 → 4
            # i=3 → 8
            dilation_size = 2 ** i

            # 最初のブロックだけnum_inputsを使用
            #
            # 2ブロック目以降は前ブロックの出力チャネル数を使用
            in_channels = (
                num_inputs
                if i == 0
                else num_channels[i - 1]
            )

            # 現在のブロックの出力チャネル数
            out_channels = num_channels[i]

            # -----------------------------------------------------------------
            # paddingの計算
            # -----------------------------------------------------------------
            #
            # padding = (kernel_size - 1) * dilation
            #
            # この値をConv1dとChomp1dに指定することで、
            # 因果性を維持しながら時系列長を保存します。
            #
            # 例：
            #
            # kernel_size=3
            # dilation=4
            #
            # padding = (3-1)*4 = 8
            #
            # Conv1dで左右に8個ずつpadding
            # Chomp1dで右端8個を削除
            #
            # 結果：
            #     入力と出力の時系列長が一致
            #
            # -----------------------------------------------------------------

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout
                )
            ]

        # 全TemporalBlockを直列接続
        self.network = nn.Sequential(*layers)

    # -------------------------------------------------------------------------
    # 5.5 順伝播
    # -------------------------------------------------------------------------
    #
    # 入力：
    #     x = (Batch, Features, Time)
    #
    # 出力：
    #     y = (Batch, LastChannels, Time)
    #
    # 各時刻の出力には、その時刻までの過去情報が反映されます。
    #
    # このクラス自体には回帰用の最終出力層はありません。
    #
    # -------------------------------------------------------------------------

    def forward(self, x):
        return self.network(x)


# =============================================================================
# ## 6. TCNの動作確認
# =============================================================================
#
# ここでは以下の多変量時系列データを想定します。
#
# Batch = 32
# Features = 6
# Time = 100
#
# 例えば6種類の特徴量：
#
#     1. Air Mass (AM)
#     2. Aerosol Optical Depth (AOD)
#     3. Total Column Water Vapor (TQV)
#     4. Total Ozone (TO3)
#     5. Angstrom Exponent
#     6. Cloud Fraction
#
# 100時刻分の気象情報から時間依存特徴を抽出します。
#
#
# ### 6.1 モデル設定
#
#     num_inputs = 6
#     num_channels = [16, 32, 64]
#     kernel_size = 3
#
# 受容野：
#
#     RF = 1 + 2*(3-1)*(2^3-1)
#        = 1 + 4*7
#        = 29
#
# 各時刻の出力は最大29時刻分の入力情報に依存します。
#
# ただし、これは理論上の受容野であり、
# 29時刻全てが予測に強く寄与するとは限りません。
#
# =============================================================================


if __name__ == "__main__":

    # 再現性のため乱数シードを固定
    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # 6.2 ダミー時系列データを作成
    # -------------------------------------------------------------------------

    batch_size = 32
    num_features = 6
    sequence_length = 100

    # 入力形状：
    # (Batch, Features, Time)
    x = torch.randn(
        batch_size,
        num_features,
        sequence_length
    )

    # -------------------------------------------------------------------------
    # 6.3 TCNモデルの構築
    # -------------------------------------------------------------------------

    model = TemporalConvNet(
        num_inputs=6,
        num_channels=[16, 32, 64],
        kernel_size=3,
        dropout=0.2
    )

    # -------------------------------------------------------------------------
    # 6.4 順伝播
    # -------------------------------------------------------------------------

    # 動作確認のため推論モードに設定
    model.eval()

    with torch.no_grad():
        y = model(x)

    print("=== TCN Shape Check ===")
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    # 想定出力：
    #
    # Input shape : torch.Size([32, 6, 100])
    # Output shape: torch.Size([32, 64, 100])

    # -------------------------------------------------------------------------
    # 6.5 Receptive Fieldの計算
    # -------------------------------------------------------------------------

    kernel_size = 3
    num_levels = 3

    receptive_field = (
        1
        + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
    )

    print("Receptive Field:", receptive_field)

    # Receptive Field: 29


# =============================================================================
# ## 7. TCNを回帰問題に利用する場合
# =============================================================================
#
# この実装のTemporalConvNetは、あくまで時系列特徴抽出器です。
#
# 出力は：
#
#     (Batch, Channels, Time)
#
# であり、直接APEなどのスカラー値を予測するものではありません。
#
# 回帰問題では、目的に応じて出力ヘッドを追加します。
#
#
# ### 7.1 Many-to-One回帰
#
# 過去の時系列から1つの目的変数を予測する場合：
#
#     Input
#       ↓
#     TCN
#       ↓
#     最終時刻の特徴ベクトルを抽出
#       ↓
#     Linear
#       ↓
#     予測値
#
# 例：
#
#     x.shape = (32, 6, 100)
#
#     TCN output = (32, 64, 100)
#
#     最終時刻を抽出：
#         h = output[:, :, -1]
#
#     h.shape = (32, 64)
#
#     Linear(64, 1)
#
#     y.shape = (32, 1)
#
# これは過去100時刻から、最終時刻に対応する
# APEを1つ予測する場合などに使用できます。
#
#
# ### 7.2 Many-to-Many回帰
#
# 各時刻のAPEをそれぞれ予測する場合：
#
#     Input
#       ↓
#     TCN
#       ↓
#     1×1 Conv1d
#       ↓
#     各時刻の予測値
#
# 例：
#
#     TCN output = (32, 64, 100)
#
#     Conv1d(64, 1, kernel_size=1)
#
#     y.shape = (32, 1, 100)
#
# この構造では、各時刻の出力が
# その時刻までの過去情報に基づいて計算されます。
#
#
# ### 7.3 未来予測への応用
#
# 最終時刻の特徴表現にLinear層を接続し、
# 将来のAPEを予測することもできます。
#
# 例：
#
#     過去24時間の気象データ
#               ↓
#              TCN
#               ↓
#       最終時刻の特徴表現
#               ↓
#            Linear
#               ↓
#       1時間後のAPE予測
#
# ただし、学習データの目的変数を
# 未来時刻に対応させる必要があります。
#
#
# =============================================================================
# ## 8. TCNとSelf-Attentionの構造的な違い
# =============================================================================
#
# ### TCN
#
# 時間方向の依存関係を畳み込みによって学習します。
#
# 特徴：
# - 時間方向に局所的なフィルタを適用する
# - dilationによって長期依存を扱う
# - 畳み込みフィルタを時刻間で共有する
# - 受容野がネットワーク構造によって決まる
# - Conv1dによって特徴量間の情報も混合される
#
#
# ### Self-Attention
#
# 各時刻間の関連性をAttention Weightによって学習します。
#
# 特徴：
# - 時刻間の関連度を入力に応じて動的に計算する
# - 通常の全時刻Attentionでは遠い時刻も直接参照できる
# - Attention Weightはサンプルによって変化する
# - 因果マスクがなければ未来情報も参照する
# - 特徴量間の相互作用も表現できる
#
#
# ### 重要な違い
#
# TCN：
#     固定された時間方向の畳み込み構造
#     ＋ 学習可能なフィルタ
#
# Self-Attention：
#     入力に応じて変化する時刻間の重み付け
#
# ただし、TCNが特徴量ごとに独立した時間依存を学習し、
# Self-Attentionが共通の時間依存だけを学習する、
# という区別は正確ではありません。
#
# どちらも多変量時系列の相互作用を学習できます。
#
# また、標準的なTCNのdilationはブロック単位で共通です。
# 特徴量ごとに異なるdilationを明示的に設定するには、
# Depthwise Conv1dや特徴量別ブランチなどの設計が必要です。
#
#
# =============================================================================
# ## 9. 本実装を利用する際の注意点
# =============================================================================
#
# 1. 入力テンソルの軸順
#
#    PyTorch Conv1d：
#        (Batch, Features, Time)
#
#    一般的な時系列データ：
#        (Batch, Time, Features)
#
#    必要に応じて：
#        x = x.transpose(1, 2)
#
#
# 2. 因果性の保証
#
#    Chomp1dによって未来時刻の参照を防ぎます。
#    ただし、前処理で未来情報を混入させないことも重要です。
#
#
# 3. 受容野の設定
#
#    時系列の長さに対して受容野が小さすぎると、
#    遠い過去の情報を利用できません。
#
#    例：
#        入力系列長 = 100
#        受容野 = 29
#
#    最終時刻の出力は直近29時刻分までしか参照できません。
#
#
# 4. Dropout
#
#    model.train()ではDropoutが有効です。
#    model.eval()ではDropoutが無効になります。
#
#
# 5. weight_norm
#
#    本実装は従来のtorch.nn.utils.weight_normを使用します。
#    新しいPyTorchではparametrizations.weight_normが推奨されます。
#
#    また、weight_norm適用後の重み初期化については、
#    内部パラメータの扱いを確認する必要があります。
#
#
# 6. 時間間隔
#
#    TCNのdilationは「何サンプル離れているか」を表します。
#
#    例えば：
#        10分間隔データのdilation=6 → 60分間隔
#        1時間間隔データのdilation=6 → 6時間間隔
#
#    時間分解能が異なるデータを組み合わせる場合、
#    時刻の整列や欠損値処理が重要になります。
#
#
# 7. このTCNは特徴抽出器
#
#    APEなどの回帰予測には別途出力ヘッドが必要です。
#    また、未来予測の場合は入力窓と予測対象時刻を
#    適切に対応させる必要があります。
#
#
# =============================================================================
# ## 10. まとめ
# =============================================================================
#
# このTCNは以下の3つのクラスから構成されています。
#
# Chomp1d
#     └─ 因果性を維持するための時間方向の切り取り
#
# TemporalBlock
#     ├─ Dilated Causal Convolution × 2
#     ├─ Weight Normalization
#     ├─ ReLU
#     ├─ Dropout
#     └─ Residual Connection
#
# TemporalConvNet
#     └─ dilationを指数的に増加させながらTemporalBlockを積層
#
#
# 最も重要なポイント：
#
# 1. Conv1dによって時間依存と特徴量間の相互作用を学習する。
#
# 2. dilationによって長期間の時間依存関係を効率的に扱う。
#
# 3. Chomp1dによって未来情報の混入を防ぐ。
#
# 4. Residual Connectionによって深いモデルの学習を助ける。
#
# 5. 最終出力は時系列特徴表現であり、回帰にはHeadが必要。
#
# =============================================================================

