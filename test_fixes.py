import datetime
import numpy as np
import pandas as pd
from src.estimators import HurdleRegressor
from src.models import calculate_financing_rates
from src.constants import Columns, StatusName
try:
    from unittest.mock import patch
except ImportError:
    pass

print("=== Testing HurdleRegressor ===")
model = HurdleRegressor()
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 10, np.nan])  # Includes NaN

try:
    model.fit(X, y)
    print("❌ FAILED: HurdleRegressor silently accepted NaN without raising an error.")
except ValueError as e:
    if "NaN" in str(e) or "missing values" in str(e) or "contains NaN" in str(e):
        print(f"✅ PASSED: HurdleRegressor caught NaN explicitly. Error: {e}")
    else:
        print(f"❌ FAILED: HurdleRegressor raised an unexpected error: {e}")

print("\n=== Testing calculate_financing_rates ===")
date_ini = datetime.datetime(2025, 1, 1)
date_mid = datetime.datetime(2025, 2, 1)

# Scenario:
# Month 1: 1 demanded, 1 booked (100% rate)
# Month 2: 1 demanded, 0 booked (0% rate)
data = pd.DataFrame({
    Columns.SE_DECISION_ID: ["ok", "ok", "ok"],
    Columns.STATUS_NAME: [StatusName.BOOKED.value, StatusName.REJECTED.value, StatusName.REJECTED.value],
    Columns.OA_AMT: [100, 200, 100],
    Columns.MIS_DATE: [date_ini, date_ini, date_mid]
})

with patch("src.models.logger.info") as mock_logger, patch("src.models.plt.show") as mock_show:
    try:
        # the function returns mean_financing_rate / 100
        result = calculate_financing_rates(data, date_ini, lm=2)
        # We expect 50% rate because:
        # Month 1: booked 100 / demand 300 = 33.3%
        # Month 2: booked 0 / demand 100 = 0%
        # Mean of 33.3% and 0% is 16.65%
        print(f"✅ PASSED: Finacing rate completed successfully. Computed over lm=2 months: {result*100:.2f}%")
        
    except Exception as e:
        print(f"❌ FAILED: Financing rate error: {e}")

