#!/usr/bin/env python3
from dataclasses import dataclass


@dataclass
class LogColumnNames:
    case_id: str = "case_id"
    activity: str = "activity"
    enable_time: str = "enable_time"
    start_time: str = "start_time"
    end_time: str = "end_time"
    resource: str = "resource"