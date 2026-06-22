"""マップ蒸留（distill_map）パッケージ。

ResNet18-FPN/Decoder Student で PatchCore Teacher の異常マップ A_T・
異常スコア s_T を dense 予測として蒸留する実験系。既存の
``src/patchcore/distill``（M0-M4: メモリバンク圧縮のスコア蒸留）とは
別軸の独立実験で、本パッケージは純粋追加。設計は
``meeting_logs/PatchCore_知識蒸留_実装設計書.md``、計画は
``meeting_logs/roadmap/roadmap_20260622.md`` を参照。
"""

from patchcore.distill_map import losses, pooling

__all__ = ["losses", "pooling"]
