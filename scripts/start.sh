#!/usr/bin/env bash
set -euo pipefail
service="${LUCEBOX_SERVICE:-lucebox.service}"
systemctl start "$service"
systemctl --no-pager status "$service"
