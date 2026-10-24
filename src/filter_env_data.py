import json
import os
import pandas as pd


def clean_5env_data():
  # Path to raw data files on local, because its to big for GitHub
  raw_path = '../raw_data/5EnvSensor_MayToDec2024_180kRows.csv.xlsx'
  output_dir = 'data'
  output_file = os.path.join(output_dir, '5EnvSensor_filtered.csv')

  # Active sensor ID to filter(Room 60 G.38 Compactus)
  filter_sensor_id = 6012002000869

  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Check if the raw data file exists (input file)
  if not os.path.exists(raw_path):
    # Fallback if running from project root
    raw_path = '5EnvSensor_MayToDec2024_180kRows.xlsx'
    if not os.path.exists(raw_path):
      print(f'[Error] Raw data file not found.')
      return

  # Print message as checkpoint
  print('Reading raw environmental dataset...')
  df_env = pd.read_excel(raw_path)

  # Filter rows matching the active sensor ID
  filter_rows = df_env[df_env['sensorid'] == filter_sensor_id]

  # For when splitting info in jsondata
  expanded_rows = []

  for idx, row in filter_rows.iterrows():
    try:
      jsondata = json.loads(row['jsondata'])

      # Info from rows to keep
      base_rows = {
          'id': row['id'],
          'sensorid': row['sensorid'],
          'devicetype': row['devicetype'],
          'status': row['status'],
          'processing_errors': row['processing_errors'],
          'createdate': row['createdate'],
          'processdate': row['processdate'],
      }

      # Extracting all variables inside jsondata
      for item in jsondata:
        var_name = item['variable']['name']
        val = item['values'][0]['value']
        # Normalize column names (eg. carbon_dioxide)
        col_key = var_name.lower().replace(' ', '_')
        base_rows[col_key] = val

      expanded_rows.append(base_rows)
    except Exception:
      continue

  # Print as checkpoint
  if not expanded_rows:
    print(f'[Warning] No records found for sensor ID {filter_sensor_id}.')
    return

  # Create final DataFrame with all columns preserved and split
  df_result = pd.DataFrame(expanded_rows)

  # Sort by createdate
  if 'createdate' in df_result.columns:
    df_result['createdate'] = pd.to_datetime(
        df_result['createdate'], errors='coerce'
    )
    df_result = df_result.sort_values('createdate').reset_index(drop=True)

  # Save to data/ folder for GitHub
  df_result.to_csv(output_file, index=False)
  print(f'Success! Filtered and flattened {len(df_result)} rows.')
  print(f'Saved with all metadata columns to: {output_file}')


if __name__ == '__main__':
  clean_5env_data()