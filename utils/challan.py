"""
Challan Manager
Handles challan creation, storage (JSON file DB), and Kaggle dataset lookup.
"""
import os, json, uuid
from datetime import datetime

try:
    import kagglehub
    import pandas as pd
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("[WARN] kagglehub not installed – plate DB lookup will be mocked")

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'challans.json')
PLATE_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'plate_db.csv')

FINE_TABLE = {
    'signal':    {'amount': 1000, 'section': 'Section 119 MV Act', 'description': 'Jumping red signal'},
    'stop_line': {'amount': 500,  'section': 'Section 122 MV Act', 'description': 'Crossing stop line'},
    'no_helmet': {'amount': 1000, 'section': 'Section 129 MV Act', 'description': 'Not wearing helmet'},
    'speeding':  {'amount': 2000, 'section': 'Section 112 MV Act', 'description': 'Over speeding'},
    'wrong_lane':{'amount': 500,  'section': 'Section 121 MV Act', 'description': 'Wrong lane driving'},
}


class ChallanManager:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._ensure_db()
        self.plate_df = None
        self._load_plate_db()

    # ── plate DB ──────────────────────────────────────────────────────────────

    def _load_plate_db(self):
        """Load Indian number plates dataset from Kaggle or local cache."""
        if os.path.exists(PLATE_DB_PATH):
            try:
                import pandas as pd
                self.plate_df = pd.read_csv(PLATE_DB_PATH)
                print(f"[INFO] Plate DB loaded from cache ({len(self.plate_df)} records)")
                return
            except Exception as e:
                print(f"[WARN] Could not load cached plate DB: {e}")

        if KAGGLE_AVAILABLE:
            try:
                from pathlib import Path
                print("[INFO] Downloading Indian Number Plates dataset from Kaggle...")
                dataset_dir = Path(kagglehub.dataset_download("dataclusterlabs/indian-number-plates-dataset"))
                tabular_exts = {'.csv', '.tsv', '.json', '.jsonl', '.parquet', '.feather', '.xlsx', '.xls'}
                tabular_files = [p for p in dataset_dir.rglob('*') if p.is_file() and p.suffix.lower() in tabular_exts]

                if not tabular_files:
                    raise RuntimeError(f"No tabular file found in downloaded dataset at {dataset_dir}")

                source_file = sorted(tabular_files)[0]
                ext = source_file.suffix.lower()
                if ext == '.csv':
                    df = pd.read_csv(source_file)
                elif ext == '.tsv':
                    df = pd.read_csv(source_file, sep='\t')
                elif ext in {'.json', '.jsonl'}:
                    df = pd.read_json(source_file, lines=(ext == '.jsonl'))
                elif ext == '.parquet':
                    df = pd.read_parquet(source_file)
                elif ext == '.feather':
                    df = pd.read_feather(source_file)
                else:
                    df = pd.read_excel(source_file)

                os.makedirs(os.path.dirname(PLATE_DB_PATH), exist_ok=True)
                df.to_csv(PLATE_DB_PATH, index=False)
                self.plate_df = df
                print(f"[INFO] Plate DB loaded from Kaggle ({len(df)} records, source: {source_file.name})")
            except Exception as e:
                print(f"[WARN] Kaggle dataset load failed: {e}")
                self._use_mock_plate_db()
        else:
            self._use_mock_plate_db()

    def _use_mock_plate_db(self):
        """Seed a small mock plate database for demo purposes."""
        try:
            import pandas as pd
            mock_data = {
                'plate_number': ['MH12AB1234','DL5CAB0001','KA01MF5678','TN22CC9999','GJ01AA0001'],
                'owner_name':   ['Raj Sharma','Priya Patel','Arjun Nair','Divya Rajan','Kiran Shah'],
                'owner_address':['Pune, MH','Delhi','Bangalore, KA','Chennai, TN','Ahmedabad, GJ'],
                'vehicle_type': ['Car','Car','Motorcycle','Car','Truck'],
                'vehicle_model':['Honda City','Swift Dzire','Bajaj Pulsar','Hyundai i20','Tata Ace'],
                'registered_on':['2019-03-15','2020-07-22','2021-01-10','2018-11-05','2022-04-30'],
            }
            self.plate_df = pd.DataFrame(mock_data)
            # Persist mock data so next startup loads cache directly
            os.makedirs(os.path.dirname(PLATE_DB_PATH), exist_ok=True)
            self.plate_df.to_csv(PLATE_DB_PATH, index=False)
            print("[INFO] Mock plate DB loaded")
        except Exception:
            self.plate_df = None

    def lookup_plate_in_db(self, plate_text: str) -> dict:
        """Return owner info for a plate number, or indicate not found."""
        if not plate_text:
            return {'found': False, 'plate': plate_text}

        clean = plate_text.upper().replace(' ', '').replace('-', '')

        if self.plate_df is not None:
            try:
                # Normalise DB column too
                col = self.plate_df.columns[0]  # first column = plate number
                mask = self.plate_df[col].astype(str).str.upper().str.replace(' ', '').str.replace('-', '') == clean
                match = self.plate_df[mask]
                if not match.empty:
                    row = match.iloc[0].to_dict()
                    return {'found': True, 'plate': plate_text, 'owner': row}
            except Exception as e:
                print(f"[WARN] DB lookup error: {e}")

        return {'found': False, 'plate': plate_text,
                'owner': {'owner_name': 'Unknown', 'owner_address': 'N/A',
                          'vehicle_type': 'Unknown', 'vehicle_model': 'Unknown'}}

    # ── challan CRUD ──────────────────────────────────────────────────────────

    def _ensure_db(self):
        if not os.path.exists(DB_PATH):
            with open(DB_PATH, 'w') as f:
                json.dump([], f)

    def _read_db(self) -> list:
        try:
            with open(DB_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def _write_db(self, data: list):
        with open(DB_PATH, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def create_challan(self, plate_text, violation_type, image_path,
                       confidence, vehicle_type, annotated_image) -> dict:
        db_info = self.lookup_plate_in_db(plate_text)
        owner = db_info.get('owner', {})
        fine_info = FINE_TABLE.get(violation_type, FINE_TABLE['signal'])

        challan = {
            'id': uuid.uuid4().hex[:10].upper(),
            'created_at': datetime.now().isoformat(),
            'plate_number': plate_text,
            'plate_found_in_db': db_info['found'],
            'owner_name': owner.get('owner_name', 'Unknown'),
            'owner_address': owner.get('owner_address', 'N/A'),
            'vehicle_type': owner.get('vehicle_type', vehicle_type),
            'vehicle_model': owner.get('vehicle_model', 'N/A'),
            'violation_type': violation_type,
            'violation_description': fine_info['description'],
            'legal_section': fine_info['section'],
            'fine_amount': fine_info['amount'],
            'status': 'Pending',
            'ocr_confidence': round(confidence * 100, 1),
            'image_path': image_path,
            'annotated_image': annotated_image,
            'issued_by': 'Automated Traffic System',
            'jurisdiction': 'Traffic Police',
        }

        data = self._read_db()
        data.insert(0, challan)
        self._write_db(data)
        return challan

    def get_all_challans(self) -> list:
        return self._read_db()

    def get_challan(self, challan_id: str) -> dict | None:
        return next((c for c in self._read_db() if c['id'] == challan_id), None)

    def get_stats(self) -> dict:
        data = self._read_db()
        if not data:
            return {'total': 0, 'pending': 0, 'paid': 0,
                    'total_fine': 0, 'by_type': {}, 'recent_7_days': 0}

        from collections import Counter
        from datetime import timedelta
        now = datetime.now()
        by_type = Counter(c['violation_type'] for c in data)
        recent = sum(1 for c in data
                     if (now - datetime.fromisoformat(c['created_at'])).days <= 7)

        return {
            'total': len(data),
            'pending': sum(1 for c in data if c['status'] == 'Pending'),
            'paid': sum(1 for c in data if c['status'] == 'Paid'),
            'total_fine': sum(c.get('fine_amount', 0) for c in data),
            'by_type': dict(by_type),
            'recent_7_days': recent,
        }
