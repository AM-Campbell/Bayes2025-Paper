import pandas as pd

def filter_hospital_data(csv_path: str, threshold: int) -> pd.DataFrame:
    """
    Filter and sort hospital data based on specific criteria.
    
    Args:
        csv_path (str): Path to the CSV file containing hospital data
        threshold (int): Threshold for sum of count_wmpcd_sig and count_pmpcd_sig
        
    Returns:
        pd.DataFrame: Filtered and sorted dataframe containing hospitals meeting criteria
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Apply filters:
    # 1. qmp > 0.05
    # 2. count_wmpcd_sig + count_pmpcd_sig > threshold
    filtered_df = df[
        (df['qmp'] > 0.05) & 
        ((df['count_wmpcd_sig'] + df['count_pmpcd_sig']) > threshold)
    ]
    
    # Create a new column for min(u_p, u_w)
    filtered_df.loc[:, 'min_utilization'] = filtered_df[['u_p', 'u_w']].min(axis=1)

    # create a column called count_pcd_sig
    filtered_df['count_pcd_sig'] = filtered_df['count_wmpcd_sig'] + filtered_df['count_pmpcd_sig']
    
    # Sort by min_utilization in descending order
    result_df = filtered_df.sort_values('min_utilization', ascending=False)
    
    # Drop the min_utilization column as it was only used for sorting
    result_df = result_df.drop('min_utilization', axis=1)

    
    return result_df

def main():
    csv_path = './1-Input/Test-Data/summaries.csv'
    threshold = 10
    result_df = filter_hospital_data(csv_path, threshold)
    print(result_df.head())

if __name__ == '__main__':
    main()