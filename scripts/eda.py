# eda.py
# author: Nicole Link, Zain Nofal, Tirth Joshi
# date 2025-12-05


import click
import os
import pandas as pd
import altair as alt


@click.command()
@click.option('--processed-training-data',type=str,required=True,help="Path to processed training CSV containing features + TYPE column.")
@click.option('--plot-to',type=str,required=True,help="Directory to save the generated EDA plots.")