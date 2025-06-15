def simple_table_line(d):
    mean = d["Total"]["scores_mean"]
    std = d["Total"]["scores_std"]

    latex = "Agent "
    for key in ["score_route", "score_penalty", "score_composed"]:
        latex += f" & {mean[key]:.1f}{{\\scriptsize\\textcolor{{gray}}{{$\pm${std[key]:.1f}}}}}"
    latex += " \\\\\\hline"
    print(latex)


def overall_results(d):
    score_order = ["score_route", "score_penalty", "score_composed"]
    # Remove the average
    average = d["Total"]
    del d["Total"]

    scenarios = sorted(d.keys())

    # For each scenario
    latex = """\
\\begin{table}[htp!]
    \centering
    \\begin{tabular}{l|c|c|c}
        \\Xhline{1pt}
        Scenario & RC $\\uparrow$ & IS $\\uparrow$ & DS $\\uparrow$\\\\\\hline\n"""

    for scenario in scenarios:
        latex += f"        {scenario}"

        mean = d[scenario]["scores_mean"]
        std = d[scenario]["scores_std"]
        for key in score_order:
            latex += f" & {mean[key]:.2f}{{\\scriptsize\\textcolor{{gray}}{{$\pm${std[key]:.2f}}}}}"

        latex += "\\\\\n"
    latex += "    \\hline\n"
    latex += "    \\textbf{Average}"
    for key in score_order:
        latex += f" & {average['scores_mean'][key]:.2f}{{\\scriptsize\\textcolor{{gray}}{{$\pm${average['scores_std'][key]:.2f}}}}}"
    latex += """\
\\\\
    \\Xhline{1pt}
    \\end{tabular}
    \\caption{Caption}
    \\label{tab:key}
\\end{table}
"""

    print()
    print()
    print(latex)
    print()
    print()


def overall_results_comparison(d1, d2):
    score_order = ["score_route", "score_penalty", "score_composed"]
    d = [d1, d2]
    # Remove the average
    average = [d1["Total"], d2["Total"]]

    scenarios = sorted(d1.keys())
    scenarios.remove("Total")

    # For each scenario
    latex = """\
\\begin{table}[h]
    \centering
    \caption{Example Table with Combined Rows in Headers}
    \setlength{\\arrayrulewidth}{0.0001pt}
    \\renewcommand{\\arraystretch}{1.2}
    \\footnotesize{
    \\begin{tabularx}{\\textwidth}{lp{0pt} XXX p{0pt} XXX}
        \\Xhline{1pt}
        \\multirow{2}{*}{Scenario} && \\multicolumn{3}{c}{\\textbf{TF++ Expert}} && \multicolumn{3}{c}{\\textbf{PDM-Lite}} \\\\
        \\cline{3-5}\\cline{7-9}
        && RC $\\uparrow$ & IS $\\uparrow$ & DS $\\uparrow$ && RC $\\uparrow$ & IS $\\uparrow$ & DS $\\uparrow$ \\\\
        \\hline
"""

    for scenario in scenarios:
        latex += f"        {scenario}"

        for i in range(2):
            latex += f"\n        &"
            mean = d[i][scenario]["scores_mean"]
            std = d[i][scenario]["scores_std"]
            for key in score_order:
                latex += f"& {mean[key]:.1f}{{\\tiny\\textcolor{{gray}}{{$\pm${std[key]:.1f}}}}} "

        latex += "\\\\\n\n"
    latex += "        \\hline\n"
    latex += "        \\textbf{Average}"
    for i in range(2):
        latex += f"\n        &"
        for key in score_order:
            latex += f"& {average[i]['scores_mean'][key]:.1f}{{\\tiny\\textcolor{{gray}}{{$\pm${average[i]['scores_std'][key]:.1f}}}}} "
    latex += "\\\\\n"
    latex += """\
        \\Xhline{1pt}
    \\end{tabularx}
    }
\\end{table}
"""

    print()
    print()
    print(latex)
    print()
    print()


import ujson

m1 = "tfpp_default"
m2 = "static-LB9s2"
path_template = "evaluation/routes_validation/{}_e30/{}_r0/results.json"

with open("evaluation/routes_validation/{}_e30/{}_r0/results.json".format(m1, m1), "r") as f:
    d1 = ujson.load(f)
with open("evaluation/routes_validation/{}_e30/{}_r0/results.json".format(m2, m2), "r") as f:
    d2 = ujson.load(f)

overall_results_comparison(d1, d2)
