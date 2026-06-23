"""Geography helpers for commitment strength scoring.

The primary path uses the city/state embedded in 247's high-school label and
offline city coordinates from geonamescache. Team locations use campus-city
overrides where available and fall back to state centroids.
"""

from __future__ import annotations

import math
import re
import zipfile
from functools import lru_cache
from pathlib import Path

try:
    import geonamescache
except ImportError:  # pragma: no cover - fallback for minimal environments.
    geonamescache = None


STATE_CENTROIDS = {
    "AL": (32.8, -86.8), "AK": (64.1, -152.3), "AZ": (34.3, -111.7),
    "AR": (35.2, -92.4), "CA": (36.8, -119.4), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7), "DE": (39.0, -75.5), "DC": (38.9, -77.0),
    "FL": (28.6, -82.4), "GA": (32.7, -83.3), "HI": (20.8, -156.3),
    "ID": (44.2, -114.6), "IL": (40.0, -89.2), "IN": (39.9, -86.3),
    "IA": (42.1, -93.5), "KS": (38.5, -98.0), "KY": (37.8, -85.8),
    "LA": (31.2, -92.3), "MA": (42.2, -71.8), "MD": (39.0, -76.7),
    "ME": (45.3, -69.0), "MI": (44.3, -85.6), "MN": (46.3, -94.2),
    "MS": (32.7, -89.7), "MO": (38.4, -92.5), "MT": (46.9, -110.4),
    "NC": (35.6, -79.4), "ND": (47.5, -100.5), "NE": (41.5, -99.8),
    "NH": (43.7, -71.6), "NJ": (40.1, -74.5), "NM": (34.4, -106.1),
    "NV": (39.3, -116.6), "NY": (42.9, -75.0), "OH": (40.4, -82.8),
    "OK": (35.6, -97.5), "OR": (44.0, -120.5), "PA": (41.2, -77.2),
    "RI": (41.7, -71.5), "SC": (33.8, -80.9), "SD": (44.4, -100.2),
    "TN": (35.8, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7),
    "VA": (37.5, -78.7), "VT": (44.0, -72.7), "WA": (47.4, -120.7),
    "WI": (44.6, -89.6), "WV": (38.6, -80.6), "WY": (43.0, -107.6),
}

GAZETTEER_ZIP = Path(__file__).resolve().parent / "data" / "geo" / "2024_Gaz_place_national.zip"


TEAM_STATES = {
    "abilene christian": "TX",
    "alabama a&m": "AL", "alabama state": "AL", "albany": "NY",
    "adams state": "CO", "air force": "CO", "akron": "OH", "alabama": "AL",
    "alcorn state": "MS", "amherst college": "MA", "appalachian state": "NC",
    "arizona": "AZ", "arizona christian": "AZ", "arizona state": "AZ",
    "arkansas": "AR", "arkansas state": "AR", "arkansas-pine bluff": "AR",
    "arkansas tech": "AR", "army": "NY", "ashland university": "OH",
    "aurora university": "IL", "austin peay": "TN", "auburn": "AL",
    "ball state": "IN", "baylor": "TX", "bemidji state": "MN", "bentley": "MA",
    "bethune-cookman": "FL",
    "black hills state": "SD",
    "boise state": "ID", "boston college": "MA", "bowling green": "OH",
    "brown": "RI", "bryant": "RI", "bucknell": "PA", "buffalo": "NY", "butler": "IN",
    "byu": "UT", "cal poly": "CA", "california": "CA", "california (pa)": "PA",
    "campbell": "NC", "carson newman": "TN",
    "carroll college": "MT", "central arkansas": "AR", "central michigan": "MI",
    "central missouri": "MO", "central state": "OH", "central washington": "WA",
    "central oklahoma": "OK", "charleston southern": "SC", "charlotte": "NC",
    "chattanooga": "TN", "chicago": "IL", "chicago state": "IL", "clarion": "PA",
    "clark atlanta": "GA",
    "cincinnati": "OH", "clemson": "SC", "coastal carolina": "SC", "colgate": "NY",
    "college of idaho": "ID", "colorado": "CO", "colorado mesa": "CO",
    "colorado school of mines": "CO",
    "colorado state": "CO", "columbia": "NY", "cornell": "NY", "csu-pueblo": "CO",
    "dakota state": "SD", "dartmouth": "NH", "davenport": "MI", "davidson": "NC",
    "dayton": "OH", "delaware": "DE",
    "delaware state": "DE", "dickinson state": "ND", "drake": "IA", "duke": "NC", "duquesne": "PA",
    "east carolina": "NC", "east tennessee state": "TN", "eastern illinois": "IL",
    "eastern kentucky": "KY", "eastern michigan": "MI", "eastern new mexico": "NM", "eastern oregon": "OR",
    "eastern washington": "WA", "east texas a&m": "TX", "faulkner": "AL", "fiu": "FL",
    "ferris state": "MI", "findlay": "OH", "florida a&m": "FL",
    "florida": "FL", "florida atlantic": "FL", "florida state": "FL",
    "fordham": "NY", "fort hays state": "KS", "fort lewis college": "CO", "fort valley state": "GA",
    "fresno state": "CA", "furman": "SC",
    "gardner-webb": "NC", "georgetown": "DC",
    "george fox": "OR", "georgetown college": "KY", "georgia": "GA", "georgia southern": "GA",
    "georgia state": "GA", "georgia tech": "GA", "grambling state": "LA",
    "grand valley state": "MI", "harvard": "MA", "hawaii": "HI", "houston": "TX",
    "gannon": "PA", "glenville state": "WV", "graceland": "IA", "hamline": "MN",
    "hampden-sydney": "VA", "hampton": "VA", "harding": "AR", "hillsdale college": "MI",
    "holy cross": "MA", "hope": "MI", "houston christian": "TX", "howard": "DC",
    "idaho": "ID", "idaho state": "ID",
    "illinois": "IL", "illinois state": "IL", "incarnate word": "TX",
    "indiana": "IN", "indiana state": "IN", "indianapolis": "IN", "iowa": "IA",
    "iowa wesleyan": "IA", "iup": "PA",
    "iowa state": "IA", "jackson state": "MS", "jacksonville state": "AL",
    "james madison": "VA", "johns hopkins": "MD", "kansas": "KS", "kansas state": "KS",
    "kent state": "OH", "kennesaw state": "GA", "kentucky": "KY",
    "kentucky wesleyan": "KY", "lafayette": "PA", "lake erie": "OH",
    "la verne": "CA", "lamar": "TX", "lane college": "TN", "lehigh": "PA",
    "lenoir-rhyne": "NC", "limestone": "SC", "lincoln (ca)": "CA", "liu": "NY",
    "liberty": "VA", "lindenwood": "MO", "louisiana": "LA", "louisiana tech": "LA",
    "louisiana-monroe": "LA", "louisville": "KY", "lsu": "LA", "marshall": "WV", "maryland": "MD",
    "maine": "ME", "marian": "IN", "mcneese": "LA", "memphis": "TN", "mercer": "GA",
    "merrimack": "MA", "mercyhurst": "PA",
    "miami": "FL", "miami (oh)": "OH",
    "michigan": "MI", "michigan state": "MI", "michigan tech": "MI", "middlebury college": "VT",
    "middle tennessee": "TN", "minnesota state mankato": "MN",
    "middle tennessee state": "TN", "minnesota": "MN",
    "mit": "MA", "minnesota duluth": "MN", "minot state": "ND", "mississippi state": "MS",
    "mississippi valley state": "MS", "missouri": "MO", "missouri state": "MO",
    "missouri southern state": "MO", "missouri western state": "MO",
    "monmouth": "NJ", "montana": "MT", "montana state": "MT",
    "montana state-northern": "MT", "montana tech": "MT",
    "morehead state": "KY", "morehouse college": "GA", "morgan state": "MD",
    "murray state": "KY", "navy": "MD", "nc state": "NC",
    "nebraska": "NE", "nevada": "NV", "new mexico": "NM", "new mexico highlands": "NM",
    "new mexico state": "NM", "newberry": "SC", "new hampshire": "NH", "nicholls": "LA", "norfolk state": "VA",
    "north alabama": "AL",
    "north carolina": "NC", "north carolina a&t": "NC", "north carolina central": "NC",
    "north dakota": "ND",
    "north dakota state": "ND", "north texas": "TX", "northern arizona": "AZ",
    "northern colorado": "CO", "northern illinois": "IL", "northern iowa": "IA",
    "north central college": "IL", "northern michigan": "MI", "northern state": "SD", "northwest missouri state": "MO",
    "northwestern state": "LA", "northwood university": "MI",
    "northwestern": "IL", "notre dame": "IN", "ohio": "OH", "ohio state": "OH",
    "ohio dominican": "OH", "oklahoma": "OK", "oklahoma baptist": "OK",
    "oklahoma state": "OK", "old dominion": "VA", "ole miss": "MS", "oregon": "OR",
    "oregon state": "OR", "penn state": "PA", "pennsylvania": "PA", "pittsburgh": "PA",
    "pittsburg state": "KS", "portland state": "OR", "presentation college": "SD",
    "princeton": "NJ", "providence": "RI",
    "pacific": "OR", "pacific lutheran": "WA", "prairie view a&m": "TX", "presbyterian": "SC", "purdue": "IN",
    "quincy": "IL",
    "rhode island": "RI", "rice": "TX", "richmond": "VA", "robert morris": "PA",
    "rocky mountain": "MT", "roosevelt": "IL", "rutgers": "NJ", "sacred heart": "CT", "sacramento state": "CA",
    "saginaw valley state": "MI", "sam houston": "TX", "san diego": "CA",
    "samford": "AL", "san diego state": "CA", "san jose state": "CA", "savannah state": "GA",
    "saint francis (pa)": "PA", "shorter": "GA", "sioux falls": "SD", "slippery rock": "PA",
    "southeast missouri state": "MO",
    "southeastern louisiana": "LA", "smu": "TX", "south alabama": "AL",
    "south carolina": "SC", "south carolina state": "SC", "south florida": "FL",
    "southeastern": "FL", "southern": "LA", "southern arkansas": "AR",
    "southern virginia": "VA",
    "southern miss": "MS",
    "south dakota": "SD", "south dakota state": "SD", "southern illinois": "IL",
    "southern oregon": "OR", "southern utah": "UT", "southwest baptist": "MO",
    "stanford": "CA", "st. augustine's": "NC",
    "st. thomas": "MN", "stetson": "FL", "stephen f. austin": "TX", "stony brook": "NY", "syracuse": "NY",
    "tarleton state": "TX", "tcu": "TX", "temple": "PA",
    "tennessee": "TN", "tennessee state": "TN", "tennessee tech": "TN",
    "texas": "TX", "texas a&m": "TX", "texas-rio grande valley": "TX",
    "texas southern": "TX", "texas state": "TX",
    "texas tech": "TX", "the citadel": "SC", "toledo": "OH", "towson": "MD",
    "tiffin": "OH", "truman state": "MO", "troy": "AL", "tufts university": "MA",
    "tusculum": "TN", "tuskegee": "AL", "tulane": "LA", "tulsa": "OK",
    "uab": "AL", "uc davis": "CA", "ucf": "FL", "ucla": "CA", "uconn": "CT",
    "ul-monroe": "LA", "umass": "MA", "unlv": "NV",
    "university of mary": "ND", "upper iowa": "IA", "usc": "CA", "ut martin": "TN",
    "utah": "UT", "utah state": "UT", "utah tech": "UT", "utep": "TX", "utsa": "TX",
    "uva wise": "VA", "valdosta state": "GA", "valparaiso": "IN",
    "vanderbilt": "TN", "villanova": "PA", "virginia": "VA", "virginia tech": "VA", "wake forest": "NC",
    "virginia union": "VA", "wagner": "NY", "walsh": "OH", "washburn": "KS",
    "washington": "WA", "washington state": "WA",
    "west alabama": "AL",
    "west florida": "FL", "west liberty": "WV", "west texas a&m": "TX",
    "west virginia": "WV", "west virginia state": "WV", "western new mexico": "NM",
    "wayne state": "MI", "weber state": "UT", "west georgia": "GA",
    "western carolina": "NC", "western colorado": "CO", "western illinois": "IL", "western kentucky": "KY",
    "western michigan": "MI", "western oregon": "OR", "wisconsin": "WI",
    "whittier college": "CA", "whitworth": "WA", "william & mary": "VA",
    "winona state": "MN", "wisconsin-platteville": "WI", "wisconsin-whitewater": "WI", "wofford": "SC",
    "wyoming": "WY", "yale": "CT", "youngstown state": "OH",
}

TEAM_CITIES = {
    "alabama": ("Tuscaloosa", "AL"), "arkansas": ("Fayetteville", "AR"),
    "auburn": ("Auburn", "AL"), "florida": ("Gainesville", "FL"),
    "georgia": ("Athens", "GA"), "kentucky": ("Lexington", "KY"),
    "lsu": ("Baton Rouge", "LA"), "mississippi state": ("Starkville", "MS"),
    "missouri": ("Columbia", "MO"), "oklahoma": ("Norman", "OK"),
    "ole miss": ("Oxford", "MS"), "south carolina": ("Columbia", "SC"),
    "tennessee": ("Knoxville", "TN"), "texas": ("Austin", "TX"),
    "texas a&m": ("College Station", "TX"), "vanderbilt": ("Nashville", "TN"),
    "illinois": ("Champaign", "IL"), "indiana": ("Bloomington", "IN"),
    "iowa": ("Iowa City", "IA"), "maryland": ("College Park", "MD"),
    "michigan": ("Ann Arbor", "MI"), "michigan state": ("East Lansing", "MI"),
    "minnesota": ("Minneapolis", "MN"), "nebraska": ("Lincoln", "NE"),
    "northwestern": ("Evanston", "IL"), "ohio state": ("Columbus", "OH"),
    "oregon": ("Eugene", "OR"), "penn state": ("State College", "PA"),
    "purdue": ("West Lafayette", "IN"), "rutgers": ("Piscataway", "NJ"),
    "ucla": ("Los Angeles", "CA"), "usc": ("Los Angeles", "CA"),
    "washington": ("Seattle", "WA"), "wisconsin": ("Madison", "WI"),
    "arizona": ("Tucson", "AZ"), "arizona state": ("Tempe", "AZ"),
    "baylor": ("Waco", "TX"), "byu": ("Provo", "UT"),
    "cincinnati": ("Cincinnati", "OH"), "colorado": ("Boulder", "CO"),
    "houston": ("Houston", "TX"), "iowa state": ("Ames", "IA"),
    "kansas": ("Lawrence", "KS"), "kansas state": ("Manhattan", "KS"),
    "oklahoma state": ("Stillwater", "OK"), "tcu": ("Fort Worth", "TX"),
    "texas tech": ("Lubbock", "TX"), "ucf": ("Orlando", "FL"),
    "utah": ("Salt Lake City", "UT"), "west virginia": ("Morgantown", "WV"),
    "boston college": ("Chestnut Hill", "MA"), "california": ("Berkeley", "CA"),
    "clemson": ("Clemson", "SC"), "duke": ("Durham", "NC"),
    "florida state": ("Tallahassee", "FL"), "georgia tech": ("Atlanta", "GA"),
    "louisville": ("Louisville", "KY"), "miami": ("Coral Gables", "FL"),
    "nc state": ("Raleigh", "NC"), "north carolina": ("Chapel Hill", "NC"),
    "pittsburgh": ("Pittsburgh", "PA"), "smu": ("Dallas", "TX"),
    "stanford": ("Stanford", "CA"), "syracuse": ("Syracuse", "NY"),
    "virginia": ("Charlottesville", "VA"), "virginia tech": ("Blacksburg", "VA"),
    "wake forest": ("Winston-Salem", "NC"), "notre dame": ("Notre Dame", "IN"),
    "boise state": ("Boise", "ID"), "memphis": ("Memphis", "TN"),
    "tulane": ("New Orleans", "LA"), "utsa": ("San Antonio", "TX"),
    "north texas": ("Denton", "TX"), "rice": ("Houston", "TX"),
    "texas state": ("San Marcos", "TX"), "utep": ("El Paso", "TX"),
}


def recruit_location_from_high_school(high_school: str) -> tuple[str, str]:
    match = re.search(r"\(([^,()]+),\s*([A-Z]{2})\)$", high_school or "")
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def recruit_state_from_high_school(high_school: str) -> str:
    return recruit_location_from_high_school(high_school)[1]


def team_state(team_key: str) -> str:
    return TEAM_STATES.get((team_key or "").lower(), "")


@lru_cache(maxsize=1)
def city_index() -> dict[tuple[str, str], tuple[float, float, int]]:
    if geonamescache is None:
        return {}
    index: dict[tuple[str, str], tuple[float, float, int]] = {}
    cities = geonamescache.GeonamesCache().get_cities().values()
    for city in cities:
        if city.get("countrycode") != "US":
            continue
        state = str(city.get("admin1code", "")).upper()
        names = [city.get("name", ""), *city.get("alternatenames", [])]
        population = int(city.get("population") or 0)
        coords = (float(city["latitude"]), float(city["longitude"]), population)
        for name in names:
            key = (str(name).strip().lower(), state)
            if not key[0] or not key[1]:
                continue
            existing = index.get(key)
            if existing is None or population > existing[2]:
                index[key] = coords
    return index


def coords_for_city_state(city: str, state: str) -> tuple[float, float] | None:
    if not city or not state:
        return None
    match = city_index().get((city.strip().lower(), state.strip().upper()))
    if match:
        return match[0], match[1]
    return census_place_index().get((city.strip().lower(), state.strip().upper()))


def normalize_place_name(name: str) -> str:
    text = re.sub(r"\s+", " ", name or "").strip()
    suffixes = [
        " consolidated government",
        " unified government",
        " metropolitan government",
        " municipality",
        " borough",
        " village",
        " town",
        " city",
        " cdp",
    ]
    lower = text.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


@lru_cache(maxsize=1)
def census_place_index() -> dict[tuple[str, str], tuple[float, float]]:
    if not GAZETTEER_ZIP.exists():
        return {}
    index: dict[tuple[str, str], tuple[float, float]] = {}
    with zipfile.ZipFile(GAZETTEER_ZIP) as archive:
        names = archive.namelist()
        if not names:
            return {}
        with archive.open(names[0]) as handle:
            header = handle.readline().decode("utf-8").strip().split("\t")
            columns = {name: idx for idx, name in enumerate(header)}
            for raw_line in handle:
                parts = raw_line.decode("utf-8", errors="ignore").strip().split("\t")
                try:
                    state = parts[columns["USPS"]].strip().upper()
                    name = parts[columns["NAME"]].strip()
                    lat = float(parts[columns["INTPTLAT"]])
                    lon = float(parts[columns["INTPTLONG"]])
                except (KeyError, IndexError, ValueError):
                    continue
                for candidate in {name, normalize_place_name(name)}:
                    key = (candidate.lower(), state)
                    if key[0] and key not in index:
                        index[key] = (lat, lon)
    return index


def coords_for_team(team_key: str) -> tuple[float, float] | None:
    key = (team_key or "").lower()
    city_state = TEAM_CITIES.get(key)
    if city_state:
        coords = coords_for_city_state(*city_state)
        if coords:
            return coords
    state = team_state(key)
    return STATE_CENTROIDS.get(state)


def miles_between_coords(home: tuple[float, float] | None, college: tuple[float, float] | None) -> int | None:
    if not home or not college:
        return None
    lat1, lon1 = map(math.radians, home)
    lat2, lon2 = map(math.radians, college)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return round(3958.8 * 2 * math.asin(math.sqrt(a)))


def miles_between_states(home_state: str, college_state: str) -> int | None:
    return miles_between_coords(STATE_CENTROIDS.get(home_state), STATE_CENTROIDS.get(college_state))


def miles_from_high_school_to_team(high_school: str, team_key: str) -> tuple[int | None, str]:
    home_city, home_state = recruit_location_from_high_school(high_school)
    home_coords = coords_for_city_state(home_city, home_state)
    precision = "city"
    if not home_coords:
        home_coords = STATE_CENTROIDS.get(home_state)
        precision = "state"
    team_coords = coords_for_team(team_key)
    miles = miles_between_coords(home_coords, team_coords)
    if miles is None:
        precision = ""
    elif precision == "city" and team_key.lower() not in TEAM_CITIES:
        precision = "city_to_state"
    return miles, precision


def distance_bucket(miles: int | None) -> str:
    if miles is None:
        return ""
    if miles == 0:
        return "in_state"
    if miles <= 250:
        return "nearby"
    if miles <= 750:
        return "regional"
    if miles <= 1500:
        return "far"
    return "cross_country"


def geography_features(high_school: str, team_key: str) -> dict[str, object]:
    home_city, home_state = recruit_location_from_high_school(high_school)
    college_state = team_state(team_key)
    home_coords = coords_for_city_state(home_city, home_state)
    college_coords = coords_for_team(team_key)
    miles, precision = miles_from_high_school_to_team(high_school, team_key)
    return {
        "home_city": home_city,
        "home_state": home_state,
        "home_lat": round(home_coords[0], 6) if home_coords else "",
        "home_lon": round(home_coords[1], 6) if home_coords else "",
        "college_state": college_state,
        "college_lat": round(college_coords[0], 6) if college_coords else "",
        "college_lon": round(college_coords[1], 6) if college_coords else "",
        "home_to_school_miles": miles if miles is not None else "",
        "distance_bucket": distance_bucket(miles),
        "geography_precision": precision,
        "is_in_state_commit": "yes" if home_state and college_state and home_state == college_state else "no" if home_state and college_state else "",
    }
