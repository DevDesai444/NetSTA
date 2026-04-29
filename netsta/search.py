"""netsta.search — CLI shim. Forwards to timingnet.search.main()."""

import sys

from timingnet.search import main


if __name__ == "__main__":
    sys.exit(main())
