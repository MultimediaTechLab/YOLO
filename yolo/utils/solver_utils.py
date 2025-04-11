from typing import Dict

import numpy as np
from rich.table import Table


def make_ap_table(score: Dict[str, float], past_result=[], max_result=None, epoch=-1):
    ap_table = Table()
    ap_table.add_column("Epoch", justify="center", style="white", width=5)
    ap_table.add_column("Avg. Precision", justify="left", style="cyan")
    ap_table.add_column("%", justify="right", style="green", width=5)
    ap_table.add_column("Avg. Recall", justify="left", style="cyan")
    ap_table.add_column("%", justify="right", style="green", width=5)

    for eps, (ap_name1, ap_color1, ap_value1, ap_name2, ap_color2, ap_value2) in past_result:
        ap_table.add_row(f"{eps: 3d}", ap_name1, f"{ap_color1}{ap_value1:.2f}", ap_name2, f"{ap_color2}{ap_value2:.2f}")
    if past_result:
        ap_table.add_row()

    color = np.where(max_result <= score, "[green]", "[red]")

    this_ap = ("AP @ .5:.95", color[0], score[0], "AP @        .5", color[1], score[1])
    metrics = [
        ("AP @ .5:.95", color[0], score[0], "AR maxDets   1", color[6], score[6]),
        ("AP @     .5", color[1], score[1], "AR maxDets  10", color[7], score[7]),
        ("AP @    .75", color[2], score[2], "AR maxDets 100", color[8], score[8]),
        ("AP  (small)", color[3], score[3], "AR     (small)", color[9], score[9]),
        ("AP (medium)", color[4], score[4], "AR    (medium)", color[10], score[10]),
        ("AP  (large)", color[5], score[5], "AR     (large)", color[11], score[11]),
    ]

    for ap_name, ap_color, ap_value, ar_name, ar_color, ar_value in metrics:
        ap_table.add_row(f"{epoch: 3d}", ap_name, f"{ap_color}{ap_value:.2f}", ar_name, f"{ar_color}{ar_value:.2f}")

    return ap_table, this_ap
