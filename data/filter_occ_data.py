import os
import pandas as pd


def filter_occupancy_data():
  raw_occ_path = '../raw_data/5occupancySensor_MayToDec2024_9MRows.csv.xlsx'
  output_dir = 'data'
  output_file = os.path.join(output_dir, 'Occupancy_filtered.csv')

  os.makedirs(output_dir, exist_ok=True)

  if not os.path.exists(raw_occ_path):
    raw_occ_path = '5occupancySensor_MayToDec2024_9MRows.csv.xlsx'
    if not os.path.exists(raw_occ_path):
      print('[Error] Raw occupancy file not found.')
      return

  print('Reading occupancy dataset...')
  xls = pd.ExcelFile(raw_occ_path)
  df_occ = pd.read_excel(raw_occ_path, sheet_name=xls.sheet_names[0])

  target_space_id = df_occ['floorspaceid'].unique()[0]

  filtered_occ = df_occ[df_occ['floorspaceid'] == target_space_id]

  filtered_occ.to_csv(output_file, index=False)
  print(
      f'Success! Filtered {len(filtered_occ)} occupancy rows for room space.'
  )
  print(f'Saved to: {output_file}')


if __name__ == '__main__':
  filter_occupancy_data()