import click
import pandas as pd


@click.group()
def cli():
    pass

@cli.command('process')
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def process(input_file, output_file):
    # cleans the data and groups it into routes
    print("Processing {}".format(input_file))

    df = pd.read_csv(
        input_file,
        usecols=['vehicle_id', 'status', 'latitude', 'longitude', 'last_modified', 'pdop', 'line', 'ziel', 'ziel_short'],
        parse_dates=['last_modified'],
    )

    without_invalid = df[(df['latitude'] != 0) & (df['longitude'] != 0) & (df['line'] != 0)]
    within_flensburg = without_invalid[
        (without_invalid['latitude'] <= 54.9)
        & (without_invalid['latitude'] >= 54.7)
        & (without_invalid['longitude'] >= 9.1)
        & (without_invalid['longitude'] <= 9.6)
    ]
    without_duplicated = within_flensburg.drop_duplicates()


    sorted = without_duplicated.sort_values(by=['ziel', 'vehicle_id', 'last_modified'], ignore_index=True)

    sorted['5min_gap'] = sorted['last_modified'].diff() > pd.to_timedelta('5min')

    sorted['vehicle_id_category'] = sorted['vehicle_id'].astype('category').cat.codes
    sorted['changed_category'] = sorted['vehicle_id_category'].diff() != 0

    sorted['split_point'] = sorted['5min_gap'] + sorted['changed_category']


    sorted['ride_id'] = (sorted['split_point'].cumsum()).astype(int)

    sorted['ride_entries'] = sorted.groupby(['ride_id']).transform('size')

    more_than_50_rides = sorted[sorted.ride_entries > 50]

    more_than_50_rides.to_csv(output_file, index=False)



@cli.command("rides")
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def create_ride_groups(input_file, output_file):
    #
    print("Processing {}".format(input_file))

    df = pd.read_csv(
        input_file,
        usecols=['vehicle_id', 'status', 'latitude', 'longitude', 'last_modified', 'pdop', 'line', 'ziel', 'ziel_short', 'ride_id'],
        parse_dates=['last_modified'],
    )

    first_last = df.groupby('ride_id').agg(['first', 'last']).reset_index()

    output_df = pd.DataFrame()
    output_df[['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']] = first_last[[('latitude', 'first'), ('longitude', 'first'), ('latitude', 'last'), ('longitude', 'last')]]
    output_df[['line', 'ziel', 'ziel_short', 'ride_id']] = first_last[[('line', 'first'), ('ziel', 'first'), ('ziel_short', 'first'), ('ride_id', '')]]

    output_df.to_csv(output_file, index=False)



if __name__ == '__main__':
    cli()