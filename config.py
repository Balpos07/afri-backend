"""Central settings. An environment variable (e.g. a Space secret) always wins;
the values below are the fallback used when no variable is set."""
import os

_DEFAULTS = {
    "DATABASE_URL": "postgresql://postgres.xfxmvqkdvykerylcgnjp:zLwWVPpKmWsX0IFj@aws-0-eu-west-1.pooler.supabase.com:5432/postgres",
    "JWT_SECRET": "sdRKCXZZ22U1E5bpRJmmh04UUGKKansqIRpcj88_j6qe0bgw-DVnB3BW73Pg9Jxr",
    "YARNGPT_API_KEY": "sk_live_nn1Pot6DEv8CxsxMJ95htsyUn9QTke22cnJExpd1BYI",
    "ALLOWED_ORIGINS": "*",
}


def get(name: str) -> str:
    return os.getenv(name) or _DEFAULTS.get(name, "")
