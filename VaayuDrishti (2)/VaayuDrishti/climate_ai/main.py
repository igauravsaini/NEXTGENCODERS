# pyre-ignore-all-errors
from fastapi import FastAPI, UploadFile, File, Request  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
import xarray as xr
# Global store for uploaded datasets (in-memory mock DB)
file_data = {}

# Set up Pyre ignores for missing standard libraries
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import pydeck as pdk  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
import os
from pydantic import BaseModel  # type: ignore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Persist active file path across restarts
ACTIVE_FILE_CACHE = os.path.join(UPLOAD_DIR, ".active_file")
app.state.active_file_path = None

def get_active_path():
    """Returns the active file path, checking disk cache if memory state is empty."""
    if app.state.active_file_path and os.path.exists(app.state.active_file_path):
        return app.state.active_file_path
    try:
        if os.path.exists(ACTIVE_FILE_CACHE):
            with open(ACTIVE_FILE_CACHE, 'r') as f:
                path = f.read().strip()
            if path and os.path.exists(path):
                app.state.active_file_path = path
                return path
    except Exception:
        pass
    return None

def save_active_path(path):
    """Saves the active file path to memory and disk."""
    app.state.active_file_path = path
    try:
        with open(ACTIVE_FILE_CACHE, 'w') as f:
            f.write(path)
    except Exception:
        pass

def find_lat_lon_keys(ds):
    """Robustly find lat/lon coordinate or dimension names in the dataset."""
    all_keys = set(list(ds.dims) + list(ds.coords))
    lat_patterns = ['lat', 'latitude', 'LAT', 'Lat', 'y', 'nav_lat', 'grid_lat', 'y_coordinate']
    lon_patterns = ['lon', 'longitude', 'LON', 'Lon', 'x', 'nav_lon', 'grid_lon', 'x_coordinate']
    
    lat_key = next((k for k in lat_patterns if k in all_keys), None)
    lon_key = next((k for k in lon_patterns if k in all_keys), None)
    return lat_key, lon_key


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/globe")
async def read_globe(request: Request):
    return templates.TemplateResponse("globe.html", {"request": request})

@app.get("/story")
async def read_story(request: Request):
    return templates.TemplateResponse("story_mode.html", {"request": request})


@app.post("/process")
async def process_data(file: UploadFile = File(...)):
    if not file.filename:
        return {"status": "error", "message": "No file provided"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    save_active_path(file_path)

    try:
        if file.filename.endswith('.nc'):
            ds = xr.open_dataset(file_path)
            variables = list(ds.data_vars)

            insight_msg = "No variables found."
            data_points = []

            if variables:
                var_name = variables[0]
                try:
                    if 'time' in ds[var_name].dims:
                        spatial_dims = [d for d in ds[var_name].dims if d != 'time']
                        time_series = ds[var_name].mean(dim=spatial_dims)
                        data_points = np.nan_to_num(np.array(time_series, dtype=float)).tolist()
                        mean_val = float(np.nanmean(time_series.values))
                        insight_msg = f"Global avg value of '{var_name}': {mean_val:.4f}"
                    else:
                        val_array = np.nan_to_num(np.array(ds[var_name].values, dtype=float).flatten())
                        data_points = val_array[:100].tolist()
                        mean_val = float(np.mean(val_array))
                        insight_msg = f"Mean value of '{var_name}': {mean_val:.4f}"
                except Exception as e:
                    insight_msg = f"Could not extract points: {str(e)}"

            var_name_returned = variables[0] if variables else "Unknown"
            lat_key, lon_key = find_lat_lon_keys(ds)
            is_spatial = lat_key is not None and lon_key is not None
            
            ds.close()
            return {
                "status": "success",
                "filename": file.filename,
                "insight": insight_msg,
                "data_points": data_points,
                "var_name": var_name_returned,
                "is_spatial": is_spatial
            }

        elif file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            insight_msg = "Not enough data for AI prediction"
            data_points = []
            col_name = df.columns[1] if len(df.columns) > 1 else df.columns[0]

            if len(df) > 1 and len(df.columns) >= 2:
                try:
                    y = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                    valid_idx = ~np.isnan(y)
                    if valid_idx.sum() > 1:
                        X = np.array(range(len(df))).reshape(-1, 1)
                        model = LinearRegression().fit(X[valid_idx], y[valid_idx])
                        prediction = model.predict(np.array([[len(df) + 5]]))[0]
                        insight_msg = f"AI Predicted [{col_name}] for +5 periods: {prediction:.2f}"
                        data_points = np.nan_to_num(y[:500]).tolist()
                    else:
                        insight_msg = "Target column has insufficient numeric data."
                except Exception as e:
                    insight_msg = f"AI Prediction Failed: {str(e)}"

            is_spatial = any(col.lower() in ['lat', 'latitude'] for col in df.columns) and \
                         any(col.lower() in ['lon', 'longitude'] for col in df.columns)

            return {
                "status": "success",
                "filename": file.filename,
                "insight": insight_msg,
                "data_points": data_points,
                "var_name": col_name,
                "is_spatial": is_spatial
            }

        else:
            return {"status": "error", "message": "Only .nc and .csv files are supported."}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/files")
async def list_files():
    """Returns a list of all uploaded files in the storage directory."""
    files = [f for f in os.listdir(UPLOAD_DIR) if not f.startswith('.')]
    return {"status": "success", "files": files}

class LocationRequest(BaseModel):
    lat: float
    lon: float


@app.post("/location")
async def process_location(loc: LocationRequest):
    file_path = get_active_path()
    if not file_path:
        return {"status": "error", "message": "No active dataset loaded. Please upload a file first."}

    try:
        if file_path.endswith('.nc'):
            ds = xr.open_dataset(file_path)
            variables = list(ds.data_vars)
            if not variables:
                ds.close()
                return {"status": "error", "message": "No variables found in dataset."}

            var_name = variables[0]

            lat_key, lon_key = find_lat_lon_keys(ds)
            if lat_key is None or lon_key is None:
                # Fallback: If no spatial indicators, return global mean for the requested metric
                spatial_dims = [d for d in ds[var_name].dims if d != 'time']
                global_series = ds[var_name].mean(dim=spatial_dims, skipna=True)
                temp_series = np.nan_to_num(np.array(global_series.values, dtype=float)).tolist()
                mean_val = float(np.nanmean(global_series.values))
                
                ds.close()
                return {
                    "status": "success",
                    "lat": loc.lat, "lon": loc.lon,
                    "is_global_fallback": True,
                    "insight": f"Dataset is not spatial. Showing global trends for '{var_name}'.",
                    "data_points": temp_series,
                    "var_name": var_name,
                    "temp": round(mean_val, 2),
                    "risk": 5.0,
                    "metrics": {"rain": 0, "wind": 0, "aqi": 50},
                    "series": {"temp": temp_series, "rain": [], "wind": [], "aqi": []},
                    "correlation": {"temp": temp_series, "rain": []},
                    "regional_dist": []
                }

            # Handle 0-360 longitude
            lon_target = loc.lon
            try:
                if float(ds[lon_key].max()) > 180 and loc.lon < 0:
                    lon_target = loc.lon + 360
            except Exception:
                pass

            try:
                sel_kwargs = {lat_key: loc.lat, lon_key: lon_target}
                local_data = ds[var_name].sel(**sel_kwargs, method="nearest")

                temp_series = []
                num_points = 1

                if 'time' in local_data.dims:
                    # Resample to annual mean if it's a time-series
                    try:
                        annual_resampled = local_data.resample(time='YE').mean()
                    except Exception:
                        try:
                            annual_resampled = local_data.resample(time='Y').mean()
                        except Exception:
                            annual_resampled = local_data
                    
                    temp_series = np.nan_to_num(np.array(annual_resampled.values, dtype=float)).tolist()
                    mean_val = float(np.nanmean(annual_resampled.values))
                    num_points = len(temp_series)
                else:
                    raw = np.nan_to_num(np.array(local_data.values, dtype=float)).flatten()
                    if len(raw) == 0:
                        ds.close()
                        return {"status": "error", "message": "No data found at this location."}
                    mean_val = float(raw[0])
                    temp_series = raw.tolist()
                    num_points = len(temp_series)

                def get_local_series(search_keys, sim_func):
                    for k in search_keys:
                        if k in ds:
                            try:
                                sd = ds[k].sel(**sel_kwargs, method="nearest")
                                if 'time' in sd.dims:
                                    return np.nan_to_num(np.array(sd.values, dtype=float)).tolist()
                                else:
                                    v = float(np.nan_to_num(np.array(sd.values, dtype=float)).flatten()[0])
                                    return [v] * num_points
                            except Exception:
                                pass
                    return [sim_func(t, i) for i, t in enumerate(temp_series)] if num_points > 1 else [sim_func(mean_val, 0)]

                rain_series = get_local_series(
                    ['prcp', 'precip', 'rain', 'tp', 'pr'],
                    lambda t, i: float(max(0.0, ((float(t) - 273.15) * 0.5 if float(t) > 250.0 else float(t) * 0.2) + float(np.random.normal(0, 2))))
                )
                rain_val = float(np.mean(rain_series)) if rain_series else 0.0

                wind_u = get_local_series(['uwnd', 'u10', 'ua'], lambda t, i: 5.0 + np.sin(i / 10.0) * 2)
                wind_v = get_local_series(['vwnd', 'v10', 'va'], lambda t, i: 3.0 + np.cos(i / 10.0) * 2)
                wind_series = [float(np.sqrt(u**2 + v**2)) for u, v in zip(wind_u, wind_v)]
                wind_speed = float(np.mean(wind_series)) if wind_series else 0.0

                aqi_series = get_local_series(
                    ['aqi', 'pm25', 'pm2p5'],
                    lambda t, i: float(min(500.0, max(20.0, abs(loc.lat) * 2.0 + float(np.random.normal(0, 5)) + (float(t) - 280.0 if float(t) > 280.0 else 0.0) * 0.5)))
                )
                aqi_val = float(np.mean(aqi_series)) if aqi_series else 0.0

                risk_score = min(max((mean_val - 10) / 3, 1), 10) if mean_val else 5.0  # type: ignore

                ds.close()

                # Generate correlation and regional data (mocked for now based on location)
                # Correlation: Temperature and Rainfall over "time"
                correlation_data = {
                    "temp": temp_series,
                    "rain": [round(float(x), 2) for x in rain_series]  # type: ignore
                }

                # Regional Distribution: Mock data for 10 regions around the globe
                regional_dist = [
                    {"region": "Arctic", "val": 20 + np.random.normal(0, 5)},
                    {"region": "Equator", "val": 80 + np.random.normal(0, 5)},
                    {"region": "Pacific", "val": 45 + np.random.normal(0, 5)},
                    {"region": "Amazon", "val": 78 + np.random.normal(0, 5)},
                    {"region": "Sahara", "val": 95 + np.random.normal(0, 5)},
                    {"region": "Himalayas", "val": 15 + np.random.normal(0, 5)},
                    {"region": "Outback", "val": 60 + np.random.normal(0, 5)},
                    {"region": "Atlantic", "val": 40 + np.random.normal(0, 5)},
                    {"region": "Siberia", "val": 10 + np.random.normal(0, 5)},
                    {"region": "Andes", "val": 30 + np.random.normal(0, 5)}
                ]

                return {
                    "status": "success",
                    "lat": loc.lat,
                    "lon": loc.lon,
                    "insight": f"Data extracted for ({loc.lat:.2f}, {loc.lon:.2f}). Variable: {var_name}.",
                    "data_points": temp_series,
                    "var_name": var_name,
                    "temp": round(float(mean_val), 2),  # type: ignore
                    "risk": round(float(risk_score), 1),  # type: ignore
                    "metrics": {
                        "rain": round(float(rain_val), 2),  # type: ignore
                        "wind": round(float(wind_speed), 1),  # type: ignore
                        "aqi": int(round(float(aqi_val)))  # type: ignore
                    },
                    "series": {
                        "temp": temp_series,
                        "rain": [round(float(x), 2) for x in rain_series],  # type: ignore
                        "wind": [round(float(x), 2) for x in wind_series],  # type: ignore
                        "aqi": [int(round(float(x))) for x in aqi_series]  # type: ignore
                    },
                    "correlation": correlation_data,
                    "regional_dist": regional_dist
                }

            except Exception as e:
                if 'ds' in locals(): ds.close()
                return {"status": "error", "message": f"Selection error: {str(e)}"}

        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude']), None)
            lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude']), None)
            
            if not lat_col or not lon_col:
                # Return global columns (first few numeric columns)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    series = np.nan_to_num(df[col].values).tolist()
                    return {
                        "status": "success",
                        "is_global_fallback": True,
                        "insight": f"CSV is not spatial. Showing trends for '{col}'.",
                        "data_points": series,
                        "var_name": col,
                        "temp": round(float(np.mean(series)), 2),
                        "risk": 4.0,
                        "metrics": {"rain": 0, "wind": 0, "aqi": 0},
                        "series": {"temp": series, "rain": [], "wind": [], "aqi": []}
                    }
                return {"status": "error", "message": "CSV contains no spatial or temporal numeric data."}

            # Simple nearest-neighbor for CSV
            df['dist'] = np.sqrt((df[lat_col] - loc.lat)**2 + (df[lon_col] - loc.lon)**2)
            closest = df.sort_values('dist').iloc[0]
            
            numeric_vals = closest.select_dtypes(include=[np.number]).to_dict()
            val = next((v for k, v in numeric_vals.items() if k not in [lat_col, lon_col, 'dist']), 0.0)
            
            return {
                "status": "success",
                "lat": float(closest[lat_col]),
                "lon": float(closest[lon_col]),
                "insight": f"Found nearest point in CSV at [{closest[lat_col]:.2f}, {closest[lon_col]:.2f}]",
                "data_points": [float(val)],
                "var_name": "CSV Data",
                "temp": float(val),
                "risk": 5.0,
                "metrics": {"rain": 0, "wind": 0, "aqi": 0}
            }
        else:
            return {"status": "error", "message": "Location querying not supported for this format."}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/year_stats")
async def get_year_stats():
    """Returns global annual mean temperatures for the thermometer."""
    file_path = get_active_path()
    if not file_path or not file_path.endswith('.nc'):
        return {"status": "success", "years": []}
    
    try:
        ds = xr.open_dataset(file_path)
        if 'time' not in ds.dims:
            ds.close()
            return {"status": "success", "years": []}
            
        variables = list(ds.data_vars)
        if not variables:
            ds.close()
            return {"status": "success", "years": []}
            
        var_name = variables[0] 
        for v in ['tas', 'temp', 't2m', 'temperature', 'air']:
            if v in ds:
                var_name = v
                break
        
        # Calculate annual global mean across all dimensions EXCEPT time
        spatial_dims = [d for d in ds[var_name].dims if d != 'time']
        annual_series = ds[var_name].mean(dim=spatial_dims, skipna=True)
        
        # Resample to yearly if needed (if it's monthly/daily)
        if hasattr(annual_series, 'resample') and len(annual_series.time) > 20: # Only resample if too many points
             try:
                 annual_series = annual_series.resample(time='YE').mean()
             except Exception:
                 try:
                     annual_series = annual_series.resample(time='Y').mean()
                 except Exception:
                     pass
             
        annual_means = annual_series.values
        # Ensure we have a 1D array
        if hasattr(annual_means, 'flatten'):
            annual_means = annual_means.flatten()
            
        time_vals = annual_series.time.values if hasattr(annual_series, 'time') else ds.time.values

        years_list = []
        for i, val_raw in enumerate(annual_means):
            val = float(val_raw)
            t_val = time_vals[i]
            # Robust year extraction
            try:
                if hasattr(t_val, 'year'):
                    year_val = getattr(t_val, 'year')
                    year_str = str(year_val)
                elif isinstance(t_val, (np.datetime64, np.ndarray)):
                    year_str = str(np.datetime_as_string(t_val, unit='Y'))
                else:
                    year_raw_str = str(t_val)
                    year_str = year_raw_str[:4]  # type: ignore
            except Exception:
                year_str = str(i)
            
            years_list.append({"year": year_str, "temp": val})
            
        ds.close()
        return {"status": "success", "years": years_list}
    except Exception as e:
        if 'ds' in locals(): ds.close()
        return {"status": "error", "message": f"Year stats failed: {str(e)}"}


@app.get("/trend_data")
async def get_trend_data():
    """Returns annual temperature and precipitation time series from the active dataset."""
    file_path = get_active_path()
    if not file_path:
        return {"status": "error", "message": "No active dataset loaded."}

    def extract_year(t_val, fallback_idx):
        try:
            if hasattr(t_val, 'year'):
                return int(t_val.year)
            elif isinstance(t_val, (np.datetime64,)):
                return int(str(np.datetime_as_string(t_val, unit='Y')))
            else:
                return int(str(t_val)[:4])
        except Exception:
            return fallback_idx

    try:
        if file_path.endswith('.nc'):
            ds = xr.open_dataset(file_path)
            variables = list(ds.data_vars)
            if not variables:
                ds.close()
                return {"status": "error", "message": "No variables in dataset."}

            # Pick temperature variable
            temp_var = variables[0]
            for v in ['tas', 'temp', 't2m', 'temperature', 'air', 'T']:
                if v in ds:
                    temp_var = v
                    break

            # Pick precipitation variable
            rain_var = None
            for v in ['pr', 'prcp', 'precip', 'tp', 'rain', 'PRCP', 'precipitation']:
                if v in ds:
                    rain_var = v
                    break

            has_time = 'time' in ds.dims

            def annual_global_mean(ds, var):
                data = ds[var]
                spatial_dims = [d for d in data.dims if d != 'time']
                series = data.mean(dim=spatial_dims, skipna=True) if spatial_dims else data
                try:
                    if len(series.time) > 20:
                        try:
                            series = series.resample(time='YE').mean()
                        except Exception:
                            try:
                                series = series.resample(time='Y').mean()
                            except Exception:
                                pass
                except Exception:
                    pass
                vals = np.nan_to_num(np.array(series.values, dtype=float)).flatten().tolist()
                try:
                    time_vals = series.time.values
                    years = [extract_year(t, i) for i, t in enumerate(time_vals[:len(vals)])]
                except Exception:
                    years = list(range(len(vals)))
                return vals, years

            if has_time:
                temp_vals, years = annual_global_mean(ds, temp_var)
                rain_vals = []
                if rain_var:
                    try:
                        rain_vals, _ = annual_global_mean(ds, rain_var)
                    except Exception:
                        rain_vals = []
            else:
                flat = np.nan_to_num(np.array(ds[temp_var].values, dtype=float).flatten())
                temp_vals = flat[:100].tolist()
                years = list(range(len(temp_vals)))
                rain_vals = []

            ds.close()
            return {
                "status": "success",
                "years": years,
                "temp": temp_vals,
                "rain": rain_vals,
                "temp_var": temp_var,
                "rain_var": rain_var or ""
            }

        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            year_col = next((c for c in df.columns if c.lower() in ['year', 'date', 'time']), None)
            temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)
            rain_col = next((c for c in df.columns if any(k in c.lower() for k in ['rain', 'precip', 'prcp'])), None)
            years = df[year_col].tolist() if year_col else list(range(len(df)))
            temp_vals = np.nan_to_num(df[temp_col].values if temp_col else df.iloc[:, 1].values).tolist()
            rain_vals = np.nan_to_num(df[rain_col].values).tolist() if rain_col else []
            return {
                "status": "success",
                "years": years,
                "temp": temp_vals,
                "rain": rain_vals,
                "temp_var": temp_col or "Value",
                "rain_var": rain_col or ""
            }

        else:
            return {"status": "error", "message": "Unsupported file format."}

    except Exception as e:
        if 'ds' in locals():
            try:
                ds.close()
            except Exception:
                pass
        return {"status": "error", "message": f"Trend data failed: {str(e)}"}


@app.get("/heatmap")
async def get_heatmap(metric: str = "temp", year_idx: int = -1):
    """Returns [lat, lon, intensity] triplets for Leaflet heatmap overlay."""
    file_path = get_active_path()
    if not file_path or not file_path.endswith('.nc'):
        return {"status": "ok", "points": []}

    try:
        ds = xr.open_dataset(file_path)
        variables = list(ds.data_vars)
        if not variables:
            ds.close()
            return {"status": "ok", "points": []}

        lat_key, lon_key = find_lat_lon_keys(ds)
        if lat_key is None or lon_key is None:
            ds.close()
            return {"status": "ok", "points": []}

        # Choose variable for the requested metric
        var_name = variables[0]
        if metric == 'rain':
            for v in ['prcp', 'precip', 'rain', 'tp', 'pr']:
                if v in ds:
                    var_name = v
                    break
        elif metric == 'wind':
            for v in ['uwnd', 'u10', 'ua', 'wspd']:
                if v in ds:
                    var_name = v
                    break
        elif metric == 'aqi':
            for v in ['aqi', 'pm25', 'pm2p5']:
                if v in ds:
                    var_name = v
                    break

        var_data = ds[var_name]

        # Average over time if present, or select specific year
        if 'time' in var_data.dims:
            if year_idx >= 0 and year_idx < len(var_data.time):
                var_data = var_data.isel(time=year_idx)
            else:
                var_data = var_data.mean(dim='time')

        var_data = var_data.squeeze()

        lats = ds[lat_key].values
        lons = ds[lon_key].values

        # Convert 0-360 lons to -180/180
        lons = np.where(lons > 180, lons - 360, lons)

        # Sample to max ~60x60 = 3600 points
        lat_step = max(1, len(lats) // 60)
        lon_step = max(1, len(lons) // 60)
        sampled_lats = lats[::lat_step]
        sampled_lons = lons[::lon_step]

        data_2d = np.nan_to_num(np.array(var_data.values, dtype=float))

        if data_2d.ndim != 2:
            ds.close()
            return {"status": "ok", "points": []}

        sampled_data = data_2d[::lat_step, ::lon_step]

        # Normalize to 0-1
        if sampled_data.size == 0:
            ds.close()
            return {"status": "ok", "points": []}

        dmin, dmax = float(sampled_data.min()), float(sampled_data.max())
        if dmax - dmin > 0:
            norm_data = (sampled_data - dmin) / (dmax - dmin)
        else:
            norm_data = np.zeros_like(sampled_data)

        points = []
        rows, cols = min(len(sampled_lats), norm_data.shape[0]), min(len(sampled_lons), norm_data.shape[1])
        for i in range(rows):
            for j in range(cols):
                val_raw = norm_data[i, j]  # type: ignore
                intensity = float(val_raw)  # type: ignore
                if intensity > 0.01:
                    l_val = float(sampled_lats[i])  # type: ignore
                    ln_val = float(sampled_lons[j])  # type: ignore
                    points.append([l_val, ln_val, round(intensity, 3)])  # type: ignore

        ds.close()
        return {"status": "ok", "points": points, "metric": metric}

    except Exception as e:
        return {"status": "error", "points": [], "message": str(e)}


@app.get("/api/3d-map")
async def get_3d_map(metric: str = 'temp', year: str = '2020'):
    file_path = get_active_path()
    if not file_path:
        return {"error": "No data available"}
    
    try:
        ds = xr.open_dataset(file_path)  # type: ignore
        
        # Map frontend metrics to standard NetCDF variables
        var_map = {
            'temp': ['tas', 't2m', 'temp', 'temperature', 'air'],
            'rain': ['pr', 'precip', 'tp', 'precipitation'],
            'wind': ['uas', 'vas', 'wind', 'u10', 'v10'],
            'aqi': ['pm25', 'aod', 'aqi', 'dust', 'co', 'o3']
        }
        
        var_name = None
        target_vars = var_map.get(metric, ['tas'])
        for v in target_vars:
            if v in ds.variables:
                var_name = v
                break
                
        if not var_name:
            var_name = list(ds.data_vars)[0]
            
        data_var = ds[var_name]
        
        # Determine coordinate names
        lat_name = next((n for n in ['lat', 'latitude', 'Y'] if n in ds.coords), None)
        lon_name = next((n for n in ['lon', 'longitude', 'X'] if n in ds.coords), None)
        time_name = next((n for n in ['time', 't', 'T'] if n in ds.coords), None)
        
        if not lat_name or not lon_name:
            return {"error": "Latitude/Longitude missing"}

        # Select data for the year
        if time_name and hasattr(data_var, 'sel'):
            try:
                # Attempt to slice by year 
                yearly_data = data_var.sel({time_name: year})
                # If multiple time steps exist in that year, take mean
                if time_name in yearly_data.dims:
                    spatial_data = yearly_data.mean(dim=time_name, skipna=True)
                else:
                    spatial_data = yearly_data
            except Exception:
                # Fallback to first time slice if year slicing fails
                spatial_data = data_var.isel({time_name: 0})
        else:
            spatial_data = data_var

        # Aggregate missing dims if it's > 2D
        dims_to_reduce = [d for d in spatial_data.dims if d not in [lat_name, lon_name]]
        if dims_to_reduce:
            spatial_data = spatial_data.mean(dim=dims_to_reduce, skipna=True)
            
        # Downsample for PyDeck performance (skip step based on data size)
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        step_lat = max(1, len(lats) // 100)
        step_lon = max(1, len(lons) // 100)
        
        sampled_lats = lats[::step_lat]
        sampled_lons = lons[::step_lon]
        grid_data = spatial_data.values[::step_lat, ::step_lon]
        
        # Normalize data to 0-1 for visual mapping
        valid_mask = ~np.isnan(grid_data)
        min_val = np.nanmin(grid_data)
        max_val = np.nanmax(grid_data)
        range_val = max_val - min_val if max_val > min_val else 1.0
        norm_data = (grid_data - min_val) / range_val
        
        points = []
        rows, cols = valid_mask.shape
        for i in range(rows):
            for j in range(cols):
                if valid_mask[i, j]:
                    val = float(norm_data[i, j])  # type: ignore
                    if val > 0.05: # Filter extremely low values to reduce points
                        points.append({
                            "lat": float(sampled_lats[i]),  # type: ignore
                            "lon": float(sampled_lons[j]),  # type: ignore
                            "weight": val
                        })
        ds.close()

        # Build PyDeck DataFrame
        df = pd.DataFrame(points)  # type: ignore
        
        # Define the 3D Hexagon Layer
        # Colors adjusted based on metric type
        color_ranges = {
            'temp': [[0,0,255],[0,255,0],[255,255,0],[255,128,0],[255,0,0]],
            'rain': [[224,247,250],[41,182,246],[2,136,209],[1,87,155],[0,30,80]],
            'wind': [[224,242,241],[38,166,154],[0,121,107],[0,77,64],[0,30,30]]
        }
        active_colors = color_ranges.get(metric, color_ranges['temp'])

        layer = pdk.Layer(
            "HexagonLayer",
            df,
            get_position=["lon", "lat"],
            auto_highlight=True,
            elevation_scale=50,
            elevation_range=[0, 3000],
            extruded=True,
            coverage=0.9,
            get_elevation_weight="weight",
            color_range=active_colors
        )

        # Set viewport to look at globe
        view_state = pdk.ViewState(
            longitude=0,
            latitude=20,
            zoom=1.5,
            min_zoom=1,
            max_zoom=15,
            pitch=40.5,
            bearing=-27.36
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style=pdk.map_styles.DARK,
            api_keys=None # Uses free base map without MapBox token
        )

        html_str = r.to_html(as_string=True)
        return {"status": "success", "html": html_str}

    except Exception as e:
        print(f"3D Map Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8001)
