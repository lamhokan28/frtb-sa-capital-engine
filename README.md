# FRTB Standardised Approach Capital Engine

This project implements a Python-based capital aggregation engine for the FRTB Standardised Approach (FRTB_SA).

The engine takes sensitivity inputs with required fields such as risk class, bucket, tenor and risk factor, then performs regulatory-style aggregation to calculate capital charges across risk classes.

## Features

- Input standardisation from Excel templates
- Delta, Vega and Curvature aggregation
- GIRR, CSR non-securitisation, Equity, FX and Commodity modules
- DRC module (CSR non-securitisation)
- Low, Medium and High correlation scenarios
- Excel output generation

## Project Structure

```text
src/        Core Python implementation
notebooks/  Demonstration notebooks
data/       Sample input files with dummy data
outputs/    Sample output files
run/        Interface file to perform capital outputs

## Disclaimer
This project is for educational and portfolio purposes only and is not intended for production regulatory reporting.
