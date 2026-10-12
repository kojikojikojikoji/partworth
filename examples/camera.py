"""End-to-end example on the public camera data from R bayesm.

    pip install "partworth[nutpie,datasets]"
    python examples/camera.py
"""

import pandas as pd

import partworth as pw
from partworth.datasets import load_camera

data = load_camera()          # downloaded from CRAN on first use
print(data)
print(data.describe())

result = pw.HierarchicalMNL(data).fit(draws=1000, tune=2000, chains=4, random_seed=1)
print(result.diagnostics().report())
print(result.population().round(3))
print(result.importances().mean().round(1))

sim = pw.MarketSimulator(result)
scenario = pd.DataFrame({
    "product": ["Canon basic", "Sony full", "Nikon mid"],
    "brand": ["canon", "sony", "nikon"],
    "pixels": ["no", "yes", "yes"],
    "zoom": ["no", "yes", "no"],
    "video": ["no", "yes", "yes"],
    "swivel": ["no", "yes", "no"],
    "wifi": ["no", "yes", "no"],
    "price": [1.29, 2.29, 1.79],          # hundreds of US dollars
})
print(sim.shares(scenario).round(3))
print(pw.wtp(result, attribute="wifi", from_level="no", to_level="yes").round(3))
