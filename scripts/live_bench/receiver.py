"""The test stack's receiver: opym-receive on its own port (see stack.sh)."""

import argparse
import logging

from opym.stream.receiver import StreamReceiver

ap = argparse.ArgumentParser()
ap.add_argument("--bind", default="tcp://127.0.0.1:5602")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
with StreamReceiver(bind_addr=args.bind) as receiver:
    receiver.run_forever()
