import pandas as pd


def df_to_latex_table(df, highlight="highest", output_file="table_output.tex"):
    """
    Convert a DataFrame to a LaTeX table with ranks in subscript and bold best values.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain metric columns, plus 'Grand Total' and 'Rank'.
    highlight : str
        'highest' or 'lowest' — controls whether higher or lower values are best.
    output_file : str
        Path to save the LaTeX output file.
    """
    metric_cols = [col for col in df.columns if col not in ["Grand Total", "Rank"]]

    # Rank: ascending=True for lowest is better, False for highest is better
    ascending = highlight == "lowest"
    ranks_df = df[metric_cols].rank(ascending=ascending, method="min")

    # Sort by overall rank
    df_sorted = df.sort_values("Rank")

    # Build LaTeX rows
    latex_rows = []
    for model, row in df_sorted.iterrows():
        row_parts = []

        # Bold model name if rank == 1
        model_display = (
            f"\\textbf{{{model}}}" if row["Rank"] == df_sorted["Rank"].min() else model
        )
        row_parts.append(model_display)

        # Format each metric with value + subscript rank
        for col in metric_cols:
            val = row[col]
            rank = int(ranks_df.loc[model, col])
            is_best = rank == 1
            formatted = (
                f"\\textbf{{{val:.3f}}}\\textsubscript{{({rank})}}"
                if is_best
                else f"{val:.3f}\\textsubscript{{({rank})}}"
            )

            row_parts.append(formatted)

        # Add Grand Total and Rank (not subscripted)
        row_parts.extend((f"{row['Grand Total']:.3f}", f"{row['Rank']}"))
        latex_rows.append(" & ".join(row_parts) + " \\\\")

    # Column headers
    header = "Method & " + " & ".join(metric_cols) + " & Grand Total & Rank \\\\"

    # Final LaTeX table
    latex_table = (
        r"""\begin{table}[htbp]
\centering
\caption{Multivariate evaluation scores. Best per column bolded, ranks in subscript.}
\begin{tabular}{l"""
        + "c" * (len(metric_cols) + 2)
        + "}\n\\toprule\n"
        + header
        + "\n\\midrule\n"
    )

    latex_table += "\n".join(latex_rows)
    latex_table += "\n\\bottomrule\n\\end{tabular}\n\\end{table}"

    # Save output
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"✅ LaTeX table saved to: {output_file}")
    return latex_table


# Usage Example
if __name__ == "__main__":
    df = pd.read_excel("results/table.xlsx")  # Replace with your file path
    df.set_index("Model", inplace=True)
    df_to_latex_table(df, highlight="highest")
