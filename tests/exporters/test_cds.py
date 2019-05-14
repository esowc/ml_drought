from cdsapi import Client # noqa
from unittest.mock import patch, Mock

from src.exporters.cds import CDSExporter, ERA5Exporter


class TestCDSExporter:

    def test_filename_year(self):

        dataset = 'megadodo-publications'
        selection_request = {
            'variable': ['towel'],
            'year': [1979, 1978, 1980]
        }

        filename = CDSExporter.make_filename(dataset, selection_request)
        expected = 'megadodo-publications_towel_1978_1980.nc'
        assert filename == expected, f'Got {filename}, expected {expected}!'

    def test_filename_date(self):
        dataset = 'megadodo-publications'
        selection_request = {
            'variable': ['towel'],
            'date': '1978-12-01/1980-12-31'
        }

        sanitized_date = selection_request["date"].replace('/', '_')
        filename = CDSExporter.make_filename(dataset, selection_request)
        expected = f'megadodo-publications_towel_{sanitized_date}.nc'
        assert filename == expected, f'Got {filename}, expected {expected}!'

    def test_selection_dict_granularity(self):

        selection_dict_monthly = ERA5Exporter.get_era5_times(granularity='monthly')
        assert 'day' not in selection_dict_monthly, 'Got day values in monthly the selection dict!'

        selection_dict_hourly = ERA5Exporter.get_era5_times(granularity='hourly')
        assert 'day' in selection_dict_hourly, 'Day values not in hourly selection dict!'

    def test_area(self):

        region = CDSExporter.get_kenya()
        kenya_str = CDSExporter.create_area(region)

        expected_str = '6.002/33.501/-5.202/42.283'
        assert kenya_str == expected_str, f'Got {kenya_str}, expected {expected_str}!'

    @patch('cdsapi.Client')
    def test_default_selection_request(self, cdsapi_mock):
        cdsapi_mock.return_value = Mock()
        exporter = ERA5Exporter()
        default_selection_request = exporter.create_selection_request('precipitation')
        expected_selection_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['precipitation'],
            'year': ['1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987',
                     '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996',
                     '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005',
                     '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
                     '2015', '2016', '2017', '2018', '2019'],
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                     '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                     '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
            'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                    '25', '26', '27', '28', '29', '30', '31'],
            'area': '6.002/33.501/-5.202/42.283'}
        for key, val in expected_selection_request.items():
            default_val = default_selection_request[key]
            assert default_val == val, f'For {key}, expected {val}, got {default_val}'

    @patch('cdsapi.Client')
    def test_user_defined_selection_requests(self, cdsapi_mock):
        cdsapi_mock.return_value = Mock()
        exporter = ERA5Exporter()

        user_defined_arguments = {
            'year': [2019],
            'day': [1],
            'month': [1],
            'time': [0]
        }
        default_selection_request = exporter.create_selection_request('precipitation',
                                                                      user_defined_arguments)
        expected_selection_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['precipitation'],
            'year': ['2019'],
            'month': ['01'],
            'time': ['00:00'],
            'day': ['01'],
            'area': '6.002/33.501/-5.202/42.283'}
        for key, val in expected_selection_request.items():
            default_val = default_selection_request[key]
            assert default_val == val, f'For {key}, expected {val}, got {default_val}'
