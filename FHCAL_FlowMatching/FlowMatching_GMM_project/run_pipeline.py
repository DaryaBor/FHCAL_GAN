#!/usr/bin/env python3
"""Single entry point, analogous to GAN_project/run_pipeline.py.

Usage: python run_pipeline.py train --help
       python run_pipeline.py evaluate --help
       python run_pipeline.py check_data --help
       python run_pipeline.py prepare_combined --help
"""
import importlib
import sys

ACTIONS = {'train':'flow_pipeline.train',
           'evaluate':'flow_pipeline.evaluate',
           'check_data':'flow_pipeline.check_data',
           'prepare_combined':'flow_pipeline.prepare_combined',
           "fit": "flow_pipeline.fit_codes",
          }

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ACTIONS:
        print('Usage: python run_pipeline.py <' + '|'.join(ACTIONS) + '> [options]')
        raise SystemExit(0 if len(sys.argv)>1 and sys.argv[1] in ('-h','--help') else 2)
    action=sys.argv.pop(1)
    module=importlib.import_module(ACTIONS[action])
    module.main()
