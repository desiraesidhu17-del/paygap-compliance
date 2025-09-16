import click
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

console = Console()

# 🔁 Reusable report logic for CLI & web
def generate_report(csv_file, save_output=False, save_chart=False):
    try:
        df = pd.read_csv(csv_file)

        # Validation
        required_columns = {'gender', 'salary'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns - set(df.columns))}")

        avg_salaries = df.groupby('gender')['salary'].mean()
        gap = abs(avg_salaries.get('Male', 0) - avg_salaries.get('Female', 0))
        gap_percent = (gap / avg_salaries.get('Male', 1)) * 100

        # Display in terminal
        console.rule(f"📊 Generating report from: {csv_file}")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Gender", justify="center")
        table.add_column("Average Salary", justify="right")

        for gender, salary in avg_salaries.items():
            table.add_row(gender, f"${salary:,.2f}")
        console.print(table)

        console.print()
        console.print(f"💰 Gender Pay Gap: ${gap:,.2f}  ({gap_percent:.2f}% lower for women)\n", style="bold yellow")

        output_path = "results.csv"
        chart_path = "chart.png"

        if save_output:
            avg_salaries.to_csv(output_path)
        if save_chart:
            avg_salaries.plot(kind="bar", color=["purple", "blue"])
            plt.title("Average Salary by Gender")
            plt.ylabel("Salary")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()

        return {
            "gender_stats": avg_salaries.to_dict(),
            "gap": gap,
            "gap_percent": gap_percent,
            "chart_path": chart_path if save_chart else None,
            "output_path": output_path if save_output else None,
        }

    except Exception as e:
        raise RuntimeError(f"❌ Error: {e}")


# 🛠️ CLI
@click.group()
def cli():
    pass

@cli.command()
@click.argument("csv_file")
@click.option("--output", is_flag=True, help="Save summary CSV")
@click.option("--chart", is_flag=True, help="Save chart image")
def report(csv_file, output, chart):
    """Generate a pay gap report from a CSV file."""
    try:
        result = generate_report(csv_file, save_output=output, save_chart=chart)
        if result.get("output_path"):
            click.echo(f"✅ Report saved to: {result['output_path']}")
        if result.get("chart_path"):
            click.echo(f"🖼️  Chart saved to: {result['chart_path']}")
    except Exception as e:
        click.echo(str(e))


@cli.command()
@click.argument("csv_file")
def summary(csv_file):
    """Print gender pay summary."""
    try:
        df = pd.read_csv(csv_file)
        male_avg = df[df["gender"] == "Male"]["salary"].mean()
        female_avg = df[df["gender"] == "Female"]["salary"].mean()
        gap = male_avg - female_avg
        percent = (gap / male_avg) * 100 if male_avg else 0

        console.print(f"\n📊 Summary for: {csv_file}\n")
        console.print(f"👥 Total employees: {len(df)}")
        console.print(f"  Female: {len(df[df['gender'] == 'Female'])}")
        console.print(f"  Male: {len(df[df['gender'] == 'Male'])}\n")
        console.print(f"💼 Avg salary overall: ${df['salary'].mean():,.2f}")
        console.print(f"\n💰 Gender Pay Gap: ${gap:,.2f} ({percent:.2f}% lower for women)\n", style="bold yellow")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")


@cli.command()
@click.argument("csv_file")
def validate(csv_file):
    """Validate CSV columns and data."""
    try:
        df = pd.read_csv(csv_file)
        required = {"employee_id", "gender", "salary"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")
        console.print(f"\n✅ File validation passed. All rows look good!\n", style="bold green")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")


# ✅ Entry point
if __name__ == "__main__":
    cli()
