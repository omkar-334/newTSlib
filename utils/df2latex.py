import pandas as pd


def df_to_latex_table(
    df, highlight="highest", output_file="table_output.tex", ours_name="Ours"
):
    """
    Convert a DataFrame to a LaTeX table with ranks in subscript and bold/underline formatting.
    Adds Venue as a subscript next to the model name.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain metric columns, 'Rank', and optionally 'Venue'.
        'Grand Total' will be ignored if present.
    highlight : str
        'highest' or 'lowest' — controls whether higher or lower values are best.
    output_file : str
        Path to save the LaTeX output file.
    ours_name : str
        Name of the method to highlight with a horizontal line after its row.
    """
    # Drop Grand Total if exists
    metric_cols = [
        col for col in df.columns if col not in ["Grand Total", "Rank", "Venue"]
    ]

    # Rank calculation
    ascending = highlight == "lowest"
    ranks_df = df[metric_cols].rank(ascending=ascending, method="min")

    # Sort by overall rank
    df_sorted = df.sort_values("Grand Total", ascending=highlight == "lowest")

    # Build LaTeX rows
    latex_rows = []
    for model, row in df_sorted.iterrows():
        row_parts = []

        # Handle model name + venue
        venue = row["Venue"] if "Venue" in df.columns and pd.notna(row["Venue"]) else ""
        if (
            highlight == "highest"
            and row["Grand Total"] == df_sorted["Grand Total"].max()
        ) or (
            highlight == "lowest"
            and row["Grand Total"] == df_sorted["Grand Total"].min()
        ):
            model_display = f"\\textbf{{{model}}}"
        else:
            model_display = model

        # Add venue as subscript if present and not our submission
        if venue and model.strip().lower() != ours_name.lower():
            model_display += f"\\,\\textsubscript{{{venue}}}"

        row_parts.append(model_display)

        # Format metrics with value + subscript rank
        for col in metric_cols:
            val = row[col]
            rank = int(ranks_df.loc[model, col])
            if rank == 1:
                formatted = f"\\textbf{{{val:.3f}}}\\textsubscript{{({rank})}}"
            elif rank == 2:
                formatted = f"\\underline{{{val:.3f}}}\\textsubscript{{({rank})}}"
            else:
                formatted = f"{val:.3f}\\textsubscript{{({rank})}}"
            row_parts.append(formatted)

        # Add Rank (keep as-is, no int conversion)
        row_parts.append(f"{row['Rank']}")

        # Build row
        row_str = " & ".join(row_parts) + " \\\\"
        latex_rows.append(row_str)

        # Insert horizontal line after ours_name
        if model.strip().lower() == ours_name.lower():
            latex_rows.append("\\midrule")

    # Column headers
    header = "Method & " + " & ".join(metric_cols) + " & Rank \\\\"

    # Tabular spec: vertical line after Method, before Rank (no outer borders)
    col_spec = "l|" + "c" * len(metric_cols) + "|c"

    # Final LaTeX table
    latex_table = (
        r"""\begin{table}[htbp]
\centering
\caption{Multivariate evaluation scores. Best per column bolded, 2nd best underlined, ranks in subscript.}
\begin{tabular}{"""
        + col_spec
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
