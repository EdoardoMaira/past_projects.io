import pandas as pd
import glob
import os

FOLDER = "/Users/edoardomaira/Desktop/nyc_taxi_project"

files = sorted(glob.glob(f"{FOLDER}/yellow_tripdata_2024-*.parquet"))

print(f"Found {len(files)} files to convert.\n")

if len(files) == 0:
    print("No files found. Check that the FOLDER path is correct.")
    exit()

total_rows = 0
converted = 0

for path in files:
    csv_name = path.replace(".parquet", ".csv")

    # skip files already converted so the script can be re-run safely
    if os.path.exists(csv_name):
        print(f"Skipping {os.path.basename(csv_name)} (already exists)")
        continue

    print(f"Converting {os.path.basename(path)} ...")
    df = pd.read_parquet(path)
    df.to_csv(csv_name, index=False)

    rows = len(df)
    total_rows += rows
    converted += 1
    print(f"  -> {os.path.basename(csv_name)} created: {rows:,} rows\n")

print("=" * 50)
print(f"Done. Files converted in this run: {converted}")
print(f"Total rows converted: {total_rows:,}")
print("=" * 50)


################################################################################


import geopandas as gpd
import matplotlib.pyplot as plt

file_path = "/Users/edoardomaira/Desktop/nyc_taxi_project/taxi_zones/taxi_zones/taxi_zones.shp"

# the companion .dbf, .prj, .shx and .cpg files are picked up automatically
print("Loading dataset...")
taxi_zones = gpd.read_file(file_path)

print("\n=== Dataset info ===")
print(taxi_zones.info())

print("\n=== First 5 rows ===")
print(taxi_zones.head())

print("\n=== Coordinate Reference System (CRS) ===")
print(taxi_zones.crs)

print("\nGenerating map...")

fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# colour each zone by its borough
taxi_zones.plot(
    column='borough',
    ax=ax,
    legend=True,
    cmap='Set3',
    edgecolor='black',
    linewidth=0.3
)

plt.title("NYC Taxi Zones by Borough", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()

# optional export to GeoJSON
# output_path = "/Users/edoardomaira/Desktop/nyc_taxi_project/taxi_zones_exported.geojson"
# taxi_zones.to_file(output_path, driver="GeoJSON")
# print(f"Exported to: {output_path}")