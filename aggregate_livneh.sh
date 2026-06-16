#!/bin/bash

LIVNEH_DIR="/net/nfs/squam/raid/data/LIVNEH/data.nodc.noaa.gov/nodc/archive/data/0129374/daily"
TMP_DIR="./tmp_livneh"
OUTPUT_FILE="livneh_monthly_reference.nc"

mkdir -p $TMP_DIR

echo "--- Starting Lightning-Fast CDO Aggregation (1980-2014) ---"

for year in {1980..2014}; do
    echo "Processing Year: $year..."
    
    # Verify that daily files exist for the year before processing
    if ls ${LIVNEH_DIR}/livneh_NAmerExt_15Oct2014.${year}*.nc >/dev/null 2>&1; then
        
        # Use -cat to pipe the wildcards as a single continuous input stream
        cdo -selname,Tmax,Tmin -monmean -cat "${LIVNEH_DIR}/livneh_NAmerExt_15Oct2014.${year}*.nc" ${TMP_DIR}/temp_${year}.nc
        cdo -selname,Prec -monsum -cat "${LIVNEH_DIR}/livneh_NAmerExt_15Oct2014.${year}*.nc" ${TMP_DIR}/prec_${year}.nc
        
        # Merge the variables back together for this year block
        cdo -O merge ${TMP_DIR}/temp_${year}.nc ${TMP_DIR}/prec_${year}.nc ${TMP_DIR}/year_${year}.nc
        rm ${TMP_DIR}/temp_${year}.nc ${TMP_DIR}/prec_${year}.nc
    fi
done

echo "Merging all years into a single timeline..."
cdo -O mergetime ${TMP_DIR}/year_*.nc $OUTPUT_FILE

echo "Cleaning up temporary scratch files..."
rm -rf $TMP_DIR

echo "Success! Final monthly validation file saved to: $OUTPUT_FILE"
