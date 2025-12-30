"""
Example: Stage 1 - Database Layer
==================================

Demonstrates how to use the NanozymeDatabase and UniProtFetcher
to build an EC -> Nanozyme function type mapping database.
"""

from nanozyme_mining.database import NanozymeDatabase, UniProtFetcher
from nanozyme_mining.utils import NanozymeType, EC_PATTERNS


def main():
    # Initialize database
    db = NanozymeDatabase(db_path="./data/nanozyme_db.sqlite")
    print("Database initialized")

    # Initialize fetcher
    fetcher = UniProtFetcher(cache_dir="./data/cache")
    print("Fetcher initialized")

    # Fetch enzymes for each nanozyme type
    for nanozyme_type, ec_list in EC_PATTERNS.items():
        print(f"\nProcessing {nanozyme_type.name}...")

        for ec_number in ec_list:
            print(f"  Fetching EC {ec_number}...")
            count = fetcher.fetch_and_populate(
                db=db,
                ec_number=ec_number,
                nanozyme_type=nanozyme_type,
                max_entries=5
            )
            print(f"  Added {count} entries")

    # Print statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total enzymes: {stats['total_enzymes']}")
    print(f"  By type: {stats['by_nanozyme_type']}")


if __name__ == "__main__":
    main()
