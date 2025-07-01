#!/usr/bin/env python3  
"""  
Copyright (c) Dish Technologies
All Rights Reserved
Confidential Property of Dish Technologies

THE MATERIAL CONTAINED HEREIN MAY ONLY BE USED SUBJECT TO AN EXECUTED
AGREEMENT BETWEEN THE USER AND DISH TECHNOLOGIES
DISCLOSURE OF THIS MATERIAL IS STRICTLY PROHIBITED EXCEPT SUBJECT TO
THE TERMS OF SUCH AN AGREEMENT.

Generalized Tile Finder for DVR Interface Navigation  

This module provides a comprehensive solution for navigating DVR interfaces to locate and select  
specific tiles based on visual templates, text labels, and content filters. It combines computer  
vision techniques, OCR processing, and intelligent grid navigation to automate tile selection  
in streaming device interfaces.  

Key Features:  
- Template matching using OpenCV for logo recognition  
- OCR-based text detection using Tesseract  
- Red highlight detection for UI navigation  
- Grid-based navigation with recovery mechanisms  
- Integrated filtering based on content information  
- REST API communication with device controllers  
- Comprehensive error handling and debugging support  

The system is designed to handle various DVR interface layouts and provides robust navigation  
with automatic recovery when the cursor moves outside expected grid boundaries.  

Author: Chris Deal
Dependencies: OpenCV, Tesseract OCR, NumPy
"""  

from __future__ import annotations    
import os    
import sys    
import time    
import uuid    
import pathlib    
import json    
import cv2    
import numpy as np    
import requests    
import pytesseract    
import re    
import subprocess    
import argparse    
import datetime  
from typing import Any, Optional, Tuple, Dict, List, Callable    
from dataclasses import dataclass    
from abc import ABC, abstractmethod    

# ─────────────────────────────────────────────────────────────────────────────    
# CONFIG & DEBUG SETUP    
# ─────────────────────────────────────────────────────────────────────────────    
SCRIPT_DIR = pathlib.Path(__file__).parent    
DEBUG_DIR = SCRIPT_DIR / "debug_output"  
DEBUG_DIR.mkdir(exist_ok=True)    
RUN_TIMESTAMP = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")  
TIMESTAMPED_DEBUG_DIR = DEBUG_DIR / RUN_TIMESTAMP  
TIMESTAMPED_DEBUG_DIR.mkdir(exist_ok=True)  
show_image = False    

YELLOW = '\033[93m'  
GREEN = '\033[92m'  
ORANGE = '\033[38;5;208m'  
RED = '\033[91m'  
RESET = '\033[0m'  

@dataclass    
class VisionConfig:    
    """  
    Configuration container for computer vision processing parameters.  

    This class holds all the tunable parameters used in image processing operations  
    including OCR preprocessing, color detection thresholds, and template matching  
    sensitivity settings.  

    Attributes:  
        ocr_scale_factor (float): Scaling factor for OCR preprocessing (default: 2.0)  
        ocr_blur_kernel (tuple): Kernel size for blur operations (default: (5, 5))  
        ocr_morph_kernel (tuple): Kernel size for morphological operations (default: (3, 3))  
        red_hue_ranges (list): HSV hue ranges for red color detection  
        red_sat_min (int): Minimum saturation for red detection (default: 150)  
        red_val_min (int): Minimum value for red detection (default: 150)  
        highlight_min_area (int): Minimum area for highlight rectangles (default: 800)  
        highlight_aspect_range (tuple): Valid aspect ratio range for highlights (default: (0.4, 2.5))  
        template_match_threshold (float): Default template matching threshold (default: 0.9)  
    """  
    ocr_scale_factor: float = 2.0    
    ocr_blur_kernel: tuple = (5, 5)    
    ocr_morph_kernel: tuple = (3, 3)    
    red_hue_ranges: list = None    
    red_sat_min: int = 150    
    red_val_min: int = 150    
    highlight_min_area: int = 800    
    highlight_aspect_range: tuple = (0.4, 2.5)    
    template_match_threshold: float = 0.9    

    def __post_init__(self):    
        """  
        Initialize default red hue ranges after object creation.  

        Sets up the HSV hue ranges that define red colors in the interface,  
        accounting for the wrap-around nature of the hue channel in HSV color space.  
        """  
        if self.red_hue_ranges is None:    
            self.red_hue_ranges = [(0, 10), (160, 180)]    

@dataclass    
class NavigationConfig:    
    """  
    Configuration container for grid-based UI navigation behavior.  

    This class defines the parameters that control how the system navigates  
    through the tile grid, including grid dimensions, retry limits, and timing delays.  

    Attributes:  
        grid_cols (int): Number of columns in the navigation grid (default: 5)  
        grid_rows (int): Number of rows in the navigation grid (default: 1)  
        max_passes (int): Maximum number of complete grid traversals (default: 3)  
        move_delay (float): Delay between navigation movements in seconds (default: 0.20)  
        capture_delay (float): Delay before capturing frames in seconds (default: 0.25)  
    """  
    grid_cols: int = 5    
    grid_rows: int = 1    
    max_passes: int = 3    
    move_delay: float = 0.20    
    capture_delay: float = 0.25    

@dataclass    
class CommandConfig:    
    """  
    Configuration container for device command mappings.  

    This class manages the mapping between logical navigation commands and the actual  
    device-specific commands sent via different communication methods (UISet vs IR).  

    Attributes:  
        uiset_commands (Dict[str, str]): Mapping of logical to UISet command names  
        ir_commands (Dict[str, str]): Mapping of logical to IR command names  
        fallback_order (List[str]): Order to try command methods when one fails  
    """  
    uiset_commands: Dict[str, str] = None    
    ir_commands: Dict[str, str] = None    
    fallback_order: List[str] = None    

    def __post_init__(self):    
        """  
        Initialize default command mappings after object creation.  

        Sets up the default mappings between logical navigation commands (UP, DOWN, etc.)  
        and the actual command strings used by UISet and IR communication methods.  
        """  
        if self.uiset_commands is None:    
            self.uiset_commands = {    
                "UP": "CmdUp",    
                "DOWN": "CmdDown",    
                "LEFT": "CmdLeft",    
                "RIGHT": "CmdRight",    
                "SELECT": "CmdSelect",    
                "OPTIONS": "CmdOptions",    
                "LIVE": "CmdLiveTV",    
                "RESET_USER_SETTINGS": "RESET_USER_SETTINGS",    
                "DVR": "CmdDVR",    
            }    

        if self.ir_commands is None:    
            self.ir_commands = {    
                "UP": "UP",    
                "DOWN": "DOWN",    
                "LEFT": "LEFT",    
                "RIGHT": "RIGHT",    
                "SELECT": "ENTER",    
                "OPTIONS": "OPTIONS",    
                "LIVE": "LIVE",    
                "DVR": "DVR",    
            }    

        if self.fallback_order is None:    
            self.fallback_order = ["ir", "uiset"]    

@dataclass    
class AppConfig:    
    """  
    Main application configuration container.  

    This is the root configuration class that aggregates all other configuration  
    components and provides application-wide settings like debug modes and API endpoints.  

    Attributes:  
        debug_enabled (bool): Whether debug mode is active (default: False)  
        debug_dir (pathlib.Path): Directory for debug output files  
        template_dir (pathlib.Path): Directory containing template images  
        studio_url (str): Base URL for Studio API communication  
        ir_sender_url (str): Base URL for IR sender API communication  
        vision (VisionConfig): Computer vision processing configuration  
        navigation (NavigationConfig): Grid navigation behavior configuration  
        commands (CommandConfig): Device command mapping configuration  
    """  
    debug_enabled: bool = False    
    debug_dir: pathlib.Path = TIMESTAMPED_DEBUG_DIR  
    template_dir: pathlib.Path = SCRIPT_DIR/"templates"    
    studio_url: str = "http://127.0.0.1:5000"    
    ir_sender_url: str = "http://127.0.0.1:5005"    
    vision: VisionConfig = None    
    navigation: NavigationConfig = None    
    commands: CommandConfig = None    

    def __post_init__(self):    
        """  
        Initialize sub-configurations and ensure debug directory exists.  

        Creates default instances of nested configuration objects if not provided  
        and ensures the debug output directory is available for use.  
        """  
        if self.vision is None:    
            self.vision = VisionConfig()    
        if self.navigation is None:    
            self.navigation = NavigationConfig()    
        if self.commands is None:    
            self.commands = CommandConfig()    
        self.debug_dir.mkdir(exist_ok=True)    

# ─────────────────────────────────────────────────────────────────────────────    
# DEBUG HELPER    
# ─────────────────────────────────────────────────────────────────────────────    

def _save_dbg(filename: str, img: np.ndarray) -> None:    
    """  
    Save debug image to timestamped debug directory.  

    This utility function saves images for debugging purposes, automatically  
    organizing them in timestamped directories to avoid conflicts between runs.  
    Optionally displays images if show_image flag is enabled.  

    Args:  
        filename (str): Name of the debug image file to save  
        img (np.ndarray): OpenCV image array to save  

    Note:  
        Images are saved to TIMESTAMPED_DEBUG_DIR with the provided filename.  
        If show_image global is True, also displays the image in a window.  
    """  
    out_path = TIMESTAMPED_DEBUG_DIR / filename    
    cv2.imwrite(str(out_path), img)    

    if show_image:    
        cv2.imshow(f"Debug: {filename}", img)    
        cv2.waitKey(1)  

# ─────────────────────────────────────────────────────────────────────────────  
#  execute_classifyfailure (on any failure)  
# ─────────────────────────────────────────────────────────────────────────────  

def execute_classifyfailure(uistephistoryid: str) -> None:  
    """  
    Execute external ClassifyFailure tool for error analysis.  

    This function invokes an external executable that performs failure classification  
    and analysis when tile finding operations fail. It's used for automated testing  
    and quality assurance purposes.  

    Args:  
        uistephistoryid (str): Unique identifier for the UI step that failed  

    Note:  
        The ClassifyFailure.exe tool is expected to be located at:  
        C:/DPUnified/Tools/ClassifyFailureAgent/ClassifyFailure.exe  

        The tool is called with arguments: ["FALSE", uistephistoryid]  
    """  
    print("execute_classifyfailure")  
    folder_path = "C:/DPUnified/Tools/ClassifyFailureAgent/"  
    try:  
        exe_path = folder_path + "ClassifyFailure.exe"  
        arguments = ["FALSE", uistephistoryid]  
        proc = subprocess.Popen(  
            [exe_path] + arguments,  
            cwd=folder_path,  
            stdout=subprocess.PIPE,  
            stderr=subprocess.PIPE  
        )  
        out, err = proc.communicate()  
        print("ClassifyFailure Output:", out.decode("utf-8"))  
        print("ClassifyFailure Error: ", err.decode("utf-8"))  
    except Exception as e:  
        print("ClassifyFailure exception:", str(e))  

# ─────────────────────────────────────────────────────────────────────────────    
# OPTIMIZED TEMPLATE MATCHING    
# ─────────────────────────────────────────────────────────────────────────────    

def _tile_has_logo_fast(tile: np.ndarray, template_path: str, threshold: float = 0.40,     
                       dbg_prefix: str = None) -> bool:    
    """  
    Perform optimized template matching with early returns and auto-scaling.  

    This function efficiently matches a template image against a tile region using  
    OpenCV template matching. It includes automatic scaling, format conversion,  
    and fallback strategies to maximize matching accuracy while maintaining performance.  

    Args:  
        tile (np.ndarray): The tile image region to search within  
        template_path (str): Path to the template image file  
        threshold (float): Confidence threshold for matching (default: 0.40)  
        dbg_prefix (str): Optional prefix for debug output files  

    Returns:  
        bool: True if template match confidence exceeds threshold, False otherwise  

    Note:  
        - Automatically handles BGRA to BGR conversion for template images  
        - Auto-scales template if it's larger than the tile region  
        - Uses grayscale matching first for speed, falls back to BGR if needed  
        - Prints confidence scores for debugging purposes  
    """  
    if tile is None or tile.size == 0:    
        return False    

    if not os.path.exists(template_path):    
        print(f"[ERROR] Template not found: {template_path}")    
        return False    

    # Load template once    
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)    
    if template is None:    
        return False    

    # Convert BGRA to BGR if needed    
    if len(template.shape) == 3 and template.shape[2] == 4:    
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)    

    # Auto-scale if needed    
    if (template.shape[0] > tile.shape[0] or template.shape[1] > tile.shape[1]):    
        scale_h = tile.shape[0] / template.shape[0]    
        scale_w = tile.shape[1] / template.shape[1]    
        scale = min(scale_h, scale_w) * 0.8    
        new_width = int(template.shape[1] * scale)    
        new_height = int(template.shape[0] * scale)    
        template = cv2.resize(template, (new_width, new_height), interpolation=cv2.INTER_AREA)    

    try:    
        # Try grayscale first (usually fastest and most reliable)    
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if len(tile.shape) == 3 else tile    
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template    

        res = cv2.matchTemplate(tile_gray, template_gray, cv2.TM_CCOEFF_NORMED)    
        _, confidence, _, _ = cv2.minMaxLoc(res)    

        if confidence >= threshold:    
            print(f"[TEMPLATE FAST] Match found: {confidence:.3f} >= {threshold}")    
            return True    

        # If grayscale fails and confidence is close, try BGR    
        if confidence > threshold * 0.8 and len(tile.shape) == 3 and len(template.shape) == 3:    
            res = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)    
            _, confidence, _, _ = cv2.minMaxLoc(res)    

            result = confidence >= threshold    
            print(f"[TEMPLATE FAST] BGR fallback: {confidence:.3f} -> {result}")    
            return result    

        print(f"[TEMPLATE FAST] No match: {confidence:.3f} < {threshold}")    
        return False    

    except Exception as e:    
        print(f"[ERROR] Template matching failed: {e}")    
        return False    

# ─────────────────────────────────────────────────────────────────────────────    
# CLI HANDLER - command line interface processing    
# ─────────────────────────────────────────────────────────────────────────────    
class CLIHandler:    
    """  
    Command line interface handler for the tile finder application.  

    This class manages the parsing and validation of command line arguments,  
    providing a comprehensive interface for configuring tile search parameters  
    including target specification, search criteria, and system configuration.  
    """  

    def __init__(self):    
        """  
        Initialize CLI handler with argument parser.  

        Creates and configures the ArgumentParser instance with all supported  
        command line options and their validation rules.  
        """  
        self.parser = self._create_parser()    

    def _create_parser(self) -> argparse.ArgumentParser:    
        """  
        Create and configure the command line argument parser.  

        Sets up all command line arguments including positional arguments for port mask,  
        optional arguments for search criteria, and configuration overrides. Includes  
        detailed help text and usage examples.  

        Returns:  
            argparse.ArgumentParser: Configured argument parser ready for use  

        Note:  
            The parser includes extensive help text with examples showing proper  
            usage patterns for different search scenarios.  
        """  
        parser = argparse.ArgumentParser(    
            description="Navigate DVR interface to find and select specified tile",    
            formatter_class=argparse.RawDescriptionHelpFormatter,    
            epilog="""    
        Examples:    
          PORT MASK = 2^[PORT_NUMBER-1]    

          # Find by template only    
          python generalized_tile_finder.py 1 --template netflix_logo    

          # Find by text only        
          python generalized_tile_finder.py 16 --label "Netflix"    

          # Find by both template AND text (both must match)    
          python generalized_tile_finder.py 4 --template netflix_logo --label"Netflix" --require-both    

          # Find by either template OR text (either can match)    
          python generalized_tile_finder.py 8 --template netflix_logo --label"Netflix"    

          # With additional filter    
          python generalized_tile_finder.py 1024 --template hulu_logo --label "Hulu" --filter "COMEDY" --template-threshold .1 --require-both  

          python generalized_tile_finder.py 1024 -t hulu_logo -l "Hulu" -f "COMEDY" --template-threshold .1 --require-both  
        """    
        )    

        # Required positional argument    
        parser.add_argument(    
            "port_mask",    
            type=int,    
            help="Port bitmask identifying target device (1, 2, 4, 8, etc.)"    
        )    

        # Target specification    
        parser.add_argument(    
            "--template", "-t",    
            help="Template image name to match on tile (without .png extension)"    
        )    

        parser.add_argument(    
            "--label", "-l",    
            help="Text to find in label below tile"    
        )    

        parser.add_argument(    
            "--require-both",    
            action="store_true",    
            help="Require both template AND label to match (default: either can match)"    
        )    

        parser.add_argument(    
            "--template-threshold",    
            type=float,    
            default=0.6,    
            help="Template matching confidence threshold (0.0-1.0)"    
        )    

        # Optional filter argument    
        parser.add_argument(    
            "-f", "--filter",    
            dest="filter_text",    
            help="Additional text filter for right-hand info pane"    
        )    

        # Configuration overrides    
        parser.add_argument(    
            "--debug",    
            action="store_true",    
            help="Enable debug mode with verbose output and image saving"    
        )    

        parser.add_argument(    
            "--template-dir",    
            type=pathlib.Path,    
            default=SCRIPT_DIR / "templates",    
            help="Directory containing template images"    
        )    

        parser.add_argument(    
            "--max-passes",    
            type=int,    
            default=3,    
            help="Maximum number of complete grid traversals"    
        )    

        parser.add_argument(    
            "--grid-size",    
            nargs=2,    
            type=int,    
            metavar=("COLS", "ROWS"),    
            help="Override grid dimensions (columns rows)"    
        )    

        return parser    

    def parse_args(self, args=None) -> argparse.Namespace:    
        """  
        Parse command line arguments with validation.  

        Processes the command line arguments and performs validation to ensure  
        they meet the requirements (e.g., port mask is a power of 2, directories exist).  

        Args:  
            args: Optional list of arguments to parse (defaults to sys.argv[1:])  

        Returns:  
            argparse.Namespace: Parsed and validated arguments  

        Raises:  
            SystemExit: If arguments are invalid or validation fails  

        Note:  
            Port mask validation ensures it's a power of 2 (1, 2, 4, 8, 16, etc.)  
            Template directory validation ensures the path exists and is accessible  
        """  
        if args is None:    
            args = sys.argv[1:]    

        parsed = self.parser.parse_args(args)    

        # Validate port mask    
        if parsed.port_mask <= 0 or (parsed.port_mask & (parsed.port_mask - 1)) != 0:    
            self.parser.error("Port mask must be a power of 2 (1, 2, 4, 8, 16, etc.)")    

        # Validate template directory    
        if not parsed.template_dir.exists():    
            self.parser.error(f"Template directory does not exist: {parsed.template_dir}")    

        return parsed    

    def create_config_from_args(self, args: argparse.Namespace) -> AppConfig:    
        """  
        Create application configuration from parsed command line arguments.  

        Converts the parsed command line arguments into a structured AppConfig  
        object, applying any overrides specified on the command line while  
        preserving defaults for unspecified options.  

        Args:  
            args (argparse.Namespace): Parsed command line arguments  

        Returns:  
            AppConfig: Complete application configuration ready for use  

        Note:  
            Command line arguments take precedence over default configuration values.  
            Grid size and max passes can be overridden via command line options.  
        """  
        config = AppConfig(    
            debug_enabled=args.debug,    
            template_dir=args.template_dir    
        )    

        # Override navigation settings if specified    
        if args.max_passes:    
            config.navigation.max_passes = args.max_passes    

        if args.grid_size:    
            config.navigation.grid_cols = args.grid_size[0]    
            config.navigation.grid_rows = args.grid_size[1]    

        return config    

# ─────────────────────────────────────────────────────────────────────────────    
# DEVICE CONTROLLER - handles both REST APIs and command mapping    
# ─────────────────────────────────────────────────────────────────────────────    
class DeviceController:    
    """  
    Device controller that communicates via REST API with command mapping.  

    This class manages communication with DVR devices through REST APIs, handling  
    both UISet commands via Studio API and IR commands via IR sender API. It provides  
    automatic fallback between communication methods and manages frame capture operations.  
    """  

    def __init__(self, config: AppConfig):    
        """  
        Initialize REST controller with API endpoints and command mapping.  

        Sets up HTTP session for efficient communication and configures API endpoints  
        and command mappings based on the provided configuration.  

        Args:  
            config (AppConfig): Application configuration containing API URLs and command mappings  

        Note:  
            Creates a persistent HTTP session with JSON content-type headers for better performance  
            across multiple API calls.  
        """  
        self.config = config    
        self.studio_url = config.studio_url    
        self.ir_sender_url = config.ir_sender_url    
        self.commands = config.commands    

        # Create persistent HTTP session for better performance    
        self.session = requests.Session()    
        self.session.headers.update({"Content-Type": "application/json"})    

    def mask_to_single_port(self, mask: int) -> int:    
        """  
        Convert a bitmask (1 << (port-1)) into the port index (1–16).  

        Transforms a port bitmask representation into a human-readable port number.  
        Used for converting between different port representation formats.  

        Args:  
            mask (int): Port bitmask with exactly one bit set  

        Returns:  
            int: Port number (1-16)  

        Raises:  
            ValueError: If mask doesn't have exactly one bit set  

        Example:  
            mask_to_single_port(1) -> 1    # Port 1  
            mask_to_single_port(4) -> 3    # Port 3  
            mask_to_single_port(8) -> 4    # Port 4  
        """  
        if mask == 0 or (mask & (mask - 1)) != 0:    
            raise ValueError("PortMask must have exactly one bit set")    
        return mask.bit_length()    

    def convertPortViewMask2String(self, mask: int) -> str:    
        """  
        Convert integer bitmask to port string format expected by API.  

        Transforms a bitmask into a string representation where each bit position  
        corresponds to a port number. Used for IR sender API communication.  

        Args:  
            mask (int): Port bitmask representing active ports  

        Returns:  
            str: String containing port numbers for active bits  

        Example:  
            convertPortViewMask2String(5) -> "13"  # Bits 0 and 2 set -> ports 1 and 3  
            convertPortViewMask2String(8) -> "4"   # Bit 3 set -> port 4  
        """  
        binary_str = format(mask, '08b')    
        reversed_binary_str = binary_str[::-1]    
        portViewString = ''.join(    
            str(idx + 1) for idx, bit in enumerate(reversed_binary_str) if bit == '1')    
        return portViewString    

    def send_command(self, port_mask: int, action: str, method: str = None) -> bool:    
        """  
        Send command using specified method, or try both if method is None.  

        Dispatches navigation commands to the target device using the specified  
        communication method (UISet or IR) or tries multiple methods in fallback order.  

        Args:  
            port_mask (int): Bitmask identifying target device port  
            action (str): Logical command name (UP, DOWN, LEFT, RIGHT, SELECT, etc.)  
            method (str, optional): Communication method ("uiset", "ir", or None for auto)  

        Returns:  
            bool: True if command was sent successfully, False otherwise  

        Note:  
            When method is None, tries all methods in the configured fallback order  
            until one succeeds or all fail. Command names are mapped to device-specific  
            commands based on the configured command mappings.  
        """  
        if method == "uiset":    
            return self._send_uiset_command(port_mask, action)    
        elif method == "ir":    
            return self._send_ir_command(port_mask, action)    
        elif method is None:    
            # Try methods in configured order    
            for try_method in self.commands.fallback_order:    
                if try_method == "uiset":    
                    if self._send_uiset_command(port_mask, action):    
                        return True    
                elif try_method == "ir":    
                    if self._send_ir_command(port_mask, action):    
                        return True    

            print(f"All methods failed for command '{action}'")    
            return False    
        else:    
            raise ValueError(f"Unknown command method: {method}")    

    def _send_uiset_command(self, port_mask: int, action: str) -> bool:    
        """  
        Send UISet command via Studio API.  

        Transmits a UISet command through the Studio REST API, handling command  
        name mapping and response validation.  

        Args:  
            port_mask (int): Bitmask identifying target device port  
            action (str): Logical command name to send  

        Returns:  
            bool: True if command was sent and acknowledged successfully, False otherwise  

        Note:  
            Maps logical command names to UISet command names using the configured  
            command mapping. Validates API response to ensure command was processed.  
        """  
        ui_name = self.commands.uiset_commands.get(action.upper(), action)    

        payload = {    
            "UISetName": ui_name,    
            "User": "",    
            "PortViews": port_mask    
        }    

        try:    
            result = self.session.post(f"{self.studio_url}/api/v1/ui-set",    
                                     json=payload, timeout=30)    
            result.raise_for_status()    
            response_data = result.json()    

            if response_data.get("ResultCode") != 0:    
                print(f"UISet '{ui_name}' failed: {response_data}")    
                return False    

            print(f"Sent UISet '{action}' -> '{ui_name}' to port {port_mask}")    
            return True    

        except Exception as e:    
            print(f"UISet command failed: {e}")    
            return False    

    def _send_ir_command(self, port_mask: int, action: str) -> bool:    
        """  
        Send IR command via IR sender.  

        Transmits an infrared command through the IR sender REST API, handling  
        port format conversion and command name mapping.  

        Args:  
            port_mask (int): Bitmask identifying target device port  
            action (str): Logical command name to send  

        Returns:  
            bool: True if command was sent successfully, False otherwise  

        Note:  
            Converts port mask to string format expected by IR sender API and  
            maps logical command names to IR command names using configured mappings.  
        """  
        command_string = self.commands.ir_commands.get(action.upper(), action)    
        portview = self.convertPortViewMask2String(port_mask)    

        url = f"{self.ir_sender_url}/api/execute-irsenderagent2"    
        payload = {    
            "portviews": portview,    
            "keyname": command_string,    
            "username": "",    
            "keypressduration": 0,    
            "delayduration": 0    
        }    

        try:    
            response = self.session.post(url, json=payload, timeout=10)    
            response.raise_for_status()    
            print(f"Sent IR '{action}' -> '{command_string}' to port {portview}")    
            return True    

        except Exception as e:    
            print(f"IR command failed: {e}")    
            return False    

    def grab_current_frame(self, mask: int) -> np.ndarray:    
        """  
        Request a snapshot via REST and return the loaded image.  

        Captures a screenshot from the specified device port using the Studio API  
        and returns it as an OpenCV image array for processing.  

        Args:  
            mask (int): Port bitmask identifying target device  

        Returns:  
            np.ndarray: OpenCV image array of the captured frame, or black frame on error  

        Note:  
            Uses a temporary file path for image storage and automatically handles  
            file cleanup. Returns a black 960x540 frame if capture fails to ensure  
            consistent return type for error handling.  
        """  
        tmp_png = fr"\\INDEVW-DEVPART03\DataStorage\Snapshots\{uuid.uuid4()}.png"    

        payload = {    
            "PortView": mask,    
            "Operation": 3,    
            "OutputFilePath": tmp_png,    
            "xLeftTopRectangle": 0, "yLeftTopRectangle": 0,    
            "RectangleWidth": 960, "RectangleHeight": 540,    
            "RecordDuration": 0, "Threshold": 0,    
            "Duration": 0, "MonitorDuration": 0,    
            "AudioCutOffThreshold": 0    
        }    

        try:    
            result = self.session.post(f"{self.studio_url}/api/v4/video", json=payload, timeout=30)    
            result.raise_for_status()    
            response_data = result.json()    

            if response_data.get("ResultCode") != 0 or not response_data.get("ResultFilePath"):    
                raise RuntimeError(f"Snapshot failed -> {response_data}")    

            img = cv2.imread(response_data["ResultFilePath"])    
            if img is None:    
                raise IOError(f"Cannot read PNG at {response_data['ResultFilePath']}")    

            if img.size == 0:    
                raise IOError(f"Loaded image is empty: {response_data['ResultFilePath']}")    

            print(f"[DEBUG] Grabbed frame: {img.shape} from {response_data['ResultFilePath']}")    
            return img    

        except Exception as e:    
            print(f"[ERROR] Failed to grab frame: {e}")    
            return np.zeros((540, 960, 3), dtype=np.uint8)    

# ─────────────────────────────────────────────────────────────────────────────    
# OPTIMIZED IMAGE PROCESSING    
# ─────────────────────────────────────────────────────────────────────────────    
class ImageProcessor:    
    """  
    Optimized image processing utilities.  

    This class provides computer vision processing capabilities optimized for  
    DVR interface analysis, including color-based highlight detection and  
    OCR preprocessing operations.  
    """  

    def __init__(self, config: VisionConfig):    
        """  
        Initialize the image processor with configuration parameters.  

        Sets up the image processor with the specified vision configuration,  
        which controls various aspects of image processing operations.  

        Args:  
            config (VisionConfig): Configuration object containing processing parameters  
        """  
        self.config = config    

    def create_red_mask_strict(self, bgr: np.ndarray) -> np.ndarray:    
        """  
        Create strict HSV mask for red highlights.  

        Generates a binary mask that identifies red-colored regions in the input image,  
        specifically tuned for detecting UI highlight rectangles in DVR interfaces.  

        Args:  
            bgr (np.ndarray): Input BGR image to process  

        Returns:  
            np.ndarray: Binary mask where white pixels indicate red regions  

        Raises:  
            ValueError: If input image is empty, None, or doesn't have 3 channels  

        Note:  
            Uses HSV color space for more robust red detection and handles the  
            wrap-around nature of hue values by combining multiple hue ranges.  
        """  
        if bgr is None or bgr.size == 0:    
            raise ValueError("Input image is empty or None for red mask creation")    

        if len(bgr.shape) != 3 or bgr.shape[2] != 3:    
            raise ValueError(f"Expected BGR image with 3 channels, got shape: {bgr.shape}")    

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)    
        masks = []    

        for hue_min, hue_max in self.config.red_hue_ranges:    
            lower = np.array([hue_min, self.config.red_sat_min, self.config.red_val_min])    
            upper = np.array([hue_max, 255, 255])    
            masks.append(cv2.inRange(hsv, lower, upper))    

        return cv2.bitwise_or(*masks) if len(masks) > 1 else masks[0]    

    def preprocess_for_ocr_fast(self, image: np.ndarray) -> np.ndarray:    
        """  
        Fast OCR preprocessing - optimized for speed.  

        Applies a series of image processing operations to enhance text readability  
        for OCR processing, balancing speed and accuracy for real-time applications.  

        Args:  
            image (np.ndarray): Input image containing text to be processed  

        Returns:  
            np.ndarray: Preprocessed binary image optimized for OCR  

        Note:  
            Processing pipeline includes: scaling, grayscale conversion, CLAHE  
            enhancement, and Otsu thresholding. Optimized for speed over maximum accuracy.  
        """  
        # Scale up for better OCR    
        scaled = cv2.resize(image, None,    
                           fx=self.config.ocr_scale_factor,    
                           fy=self.config.ocr_scale_factor,    
                           interpolation=cv2.INTER_CUBIC)    

        # Convert to grayscale    
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)    

        # Simple CLAHE    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
        enhanced = clahe.apply(gray)    

        # Otsu thresholding    
        _, binary = cv2.threshold(enhanced, 0, 255,    
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)    

        return binary    

# ─────────────────────────────────────────────────────────────────────────────    
# OPTIMIZED HIGHLIGHT DETECTOR    
# ─────────────────────────────────────────────────────────────────────────────    
class HighlightDetector:    
    """  
    Optimized detector for UI highlight rectangles.  

    This class specializes in detecting red highlight rectangles that indicate  
    the currently selected tile in DVR interfaces, using color-based detection  
    and geometric filtering to identify valid highlights.  
    """  

    def __init__(self, processor: ImageProcessor):    
        """  
        Initialize highlight detector with image processor.  

        Sets up the detector with a reference to an ImageProcessor instance  
        for performing color-based detection operations.  

        Args:  
            processor (ImageProcessor): Image processor instance for color detection  
        """  
        self.processor = processor    

    def find_highlight_rect_strict(self, frame: np.ndarray, debug_prefix: str = "") -> Optional[Tuple[int, int, int, int]]:    
        """  
        Locate the red highlight rectangle - optimized version.  

        Detects red highlight rectangles in the input frame using color-based  
        segmentation and geometric filtering to identify the most likely highlight.  

        Args:  
            frame (np.ndarray): Input frame to search for highlights  
            debug_prefix (str): Optional prefix for debug image filenames  

        Returns:  
            Optional[Tuple[int, int, int, int]]: Bounding box (x, y, width, height) of  
            the best highlight found, or None if no valid highlight detected  

        Note:  
            Uses area and aspect ratio filtering to distinguish highlights from  
            other red elements in the interface. Saves debug images when prefix provided.  
        """  
        if frame is None or frame.size == 0:    
            return None    

        if len(frame.shape) != 3:    
            return None    

        try:    
            mask = self.processor.create_red_mask_strict(frame)    
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

            best_rect = None    
            best_area = 0    

            for c in cnts:    
                x, y, w, h = cv2.boundingRect(c)    
                area = cv2.contourArea(c)    
                aspect = (w / h) if h else 0    

                if (area > self.processor.config.highlight_min_area and    
                    self.processor.config.highlight_aspect_range[0] < aspect < self.processor.config.highlight_aspect_range[1]):    
                    if area > best_area:    
                        best_area = area    
                        best_rect = (x, y, w, h)    

            if best_rect and debug_prefix:    
                x, y, w, h = best_rect    
                dbg_img = frame.copy()    
                cv2.rectangle(dbg_img, (x, y), (x + w, y + h), (0, 255, 0), 3)    
                _save_dbg(f"{debug_prefix}_highlight.png", dbg_img)    

            return best_rect    

        except Exception as e:    
            print(f"[ERROR] Exception in highlight detection: {e}")    
            return None    

# ─────────────────────────────────────────────────────────────────────────────    
# OPTIMIZED OCR ENGINE    
# ─────────────────────────────────────────────────────────────────────────────    
class OCREngine:    
    """  
    Optimized OCR engine with smart fallbacks.  

    This class provides text recognition capabilities optimized for DVR interface  
    text, with multiple preprocessing strategies and OCR configurations to handle  
    various text rendering scenarios and quality levels.  
    """  

    def __init__(self, processor: ImageProcessor):    
        """  
        Initialize OCR engine with image processor.  

        Sets up the OCR engine with a reference to an ImageProcessor for  
        performing text preprocessing operations.  

        Args:  
            processor (ImageProcessor): Image processor instance for preprocessing  
        """  
        self.processor = processor    

    def quick_text_check(self, roi: np.ndarray, target_text: str) -> bool:    
        """  
        Fast text check - single OCR attempt with best settings.  

        Performs a rapid text recognition check using optimized preprocessing  
        and OCR settings for speed-critical applications.  

        Args:  
            roi (np.ndarray): Region of interest containing text to recognize  
            target_text (str): Text string to search for in the OCR results  

        Returns:  
            bool: True if target text is found in the OCR results, False otherwise  

        Note:  
            Uses single-line PSM mode and character whitelist for improved speed  
            and accuracy. Performs case-insensitive matching with word-level fallback.  
        """  
        if roi.size == 0:    
            return False    

        try:    
            # Fast preprocessing    
            processed = self.processor.preprocess_for_ocr_fast(roi)    

            # Single OCR attempt with optimized config    
            config = "--oem 3 --psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -c user_defined_dpi=300"    
            text = pytesseract.image_to_string(processed, config=config).strip()    

            if not text:    
                return False    

            # Simple case-insensitive matching    
            text_upper = text.upper()    
            target_upper = target_text.upper()    

            # Check if target is in extracted text    
            if target_upper in text_upper:    
                print(f"[QUICK OCR] Found '{target_text}' in '{text}'")    
                return True    

            # Check word-by-word    
            text_words = text_upper.split()    
            target_words = target_upper.split()    

            for target_word in target_words:    
                if any(target_word in text_word for text_word in text_words):    
                    print(f"[QUICK OCR] Found word '{target_word}' in text")    
                    return True    

            return False    

        except Exception as e:    
            print(f"[QUICK OCR] Failed: {e}")    
            return False    

    def comprehensive_text_check(self, roi: np.ndarray, target_text: str, dbg_prefix: str = None) -> bool:    
        """  
        Comprehensive text check - multiple methods, but sequential.  

        Performs thorough text recognition using multiple preprocessing approaches  
        and OCR configurations to maximize detection accuracy for difficult text.  

        Args:  
            roi (np.ndarray): Region of interest containing text to recognize  
            target_text (str): Text string to search for in the OCR results  
            dbg_prefix (str): Optional prefix for debug image filenames  

        Returns:  
            bool: True if target text is found using any method, False otherwise  

        Note:  
            Tries multiple preprocessing methods (original, processed, grayscale, enhanced)  
            combined with different OCR configurations (single line, single word, default).  
            Uses flexible text matching to handle OCR errors and variations.  
        """  
        if roi.size == 0:    
            return False    

        print(f"[COMPREHENSIVE OCR] Checking for '{target_text}'")    

        # Try different preprocessing approaches in order of likelihood    
        preprocessing_methods = [    
            ("original", lambda img: img),    
            ("processed", lambda img: self.processor.preprocess_for_ocr_fast(img)),    
            ("grayscale", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img),    
            ("enhanced", lambda img: self._enhance_contrast(img)),    
        ]    

        # Try different OCR configs in order of speed/reliability    
        ocr_configs = [    
            {"name": "single_line", "config": "--oem 3 --psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"},    
            {"name": "single_word", "config": "--oem 3 --psm 8 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"},    
            {"name": "default", "config": "--oem 3 --psm 6"},    
        ]    

        target_words = target_text.upper().split()    

        for preprocess_name, preprocess_func in preprocessing_methods:    
            try:    
                processed_img = preprocess_func(roi)    

                if dbg_prefix:    
                    _save_dbg(f"{dbg_prefix}_preprocess_{preprocess_name}.png", processed_img)    

                for ocr_config in ocr_configs:    
                    try:    
                        extracted_text = pytesseract.image_to_string(processed_img, config=ocr_config["config"]).strip()    

                        if extracted_text:    
                            combination = f"{preprocess_name}+{ocr_config['name']}"    
                            print(f"[COMPREHENSIVE OCR] {combination}: '{extracted_text}'")    

                            # Check if this extraction contains our target    
                            if self._flexible_text_match(extracted_text, target_words):    
                                print(f"[COMPREHENSIVE OCR] ✓ MATCH found with {combination}")    
                                return True    

                    except Exception as e:    
                        print(f"[COMPREHENSIVE OCR] {preprocess_name}+{ocr_config['name']} failed: {e}")    
                        continue    

            except Exception as e:    
                print(f"[COMPREHENSIVE OCR] Preprocessing {preprocess_name} failed: {e}")      
                continue      

        return False      

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:      
        """    
        Enhance contrast for difficult text.    

        Applies contrast enhancement techniques specifically designed for improving    
        text readability in low-contrast or poorly rendered text regions.    

        Args:    
            img (np.ndarray): Input image to enhance (BGR or grayscale)    

        Returns:    
            np.ndarray: Contrast-enhanced grayscale image    

        Note:    
            Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) followed    
            by bilateral filtering to enhance text while reducing noise.    
        """    
        if len(img.shape) == 3:      
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
        else:      
            gray = img.copy()      

        # CLAHE      
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))      
        enhanced = clahe.apply(gray)      

        # Bilateral filter      
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)      

        return enhanced      

    def _flexible_text_match(self, extracted_text: str, target_words: List[str]) -> bool:      
        """    
        Flexible text matching with multiple strategies.    

        Implements multiple text matching strategies to handle OCR errors,    
        formatting variations, and partial matches that commonly occur    
        in automated text recognition scenarios.    

        Args:    
            extracted_text (str): Text extracted from OCR processing    
            target_words (List[str]): List of target words to search for    

        Returns:    
            bool: True if target words are found using any matching strategy    

        Note:    
            Tries exact word matching, substring matching, and complete phrase    
            matching to maximize detection accuracy while minimizing false positives.    
        """    
        if not extracted_text.strip():      
            return False      

        # Clean extracted text      
        extracted_clean = re.sub(r'[^A-Za-z0-9\s]', ' ', extracted_text)      
        extracted_upper = extracted_clean.upper()      
        extracted_words = extracted_upper.split()      

        # Strategy 1: All target words found as exact words      
        all_words_found = all(word in extracted_words for word in target_words)      
        if all_words_found:      
            return True      

        # Strategy 2: All target words found as substrings      
        all_substrings_found = all(word in extracted_upper for word in target_words)      
        if all_substrings_found:      
            return True      

        # Strategy 3: Target text as complete substring      
        target_complete = ' '.join(target_words)      
        if target_complete in extracted_upper:      
            return True      

        return False      

    def quick_filter_check(self, frame: np.ndarray, filter_text: str) -> bool:      
        """    
        Quick filter check in right pane - optimized for speed with enhanced debugging.    

        Performs rapid text detection in the right information pane of the interface,    
        checking multiple likely locations where filter information typically appears.    

        Args:    
            frame (np.ndarray): Full interface frame to search within    
            filter_text (str): Filter text to search for (e.g., "COMEDY", "ACTION")    

        Returns:    
            bool: True if filter text is found in any checked region, False otherwise    

        Note:    
            Focuses on right pane regions where content information is typically    
            displayed. Uses both quick and comprehensive OCR methods for reliability.    
        """    
        h, w = frame.shape[:2]      

        print(f"[FILTER DEBUG] Frame size: {w}x{h}, looking for: '{filter_text}'")      

        # Check multiple locations - not just right pane      
        filter_regions = [      
            # Right pane regions (original)      
            (int(0.75*w), int(0.20*h), w, int(0.45*h)),  # Top of right pane      
            (int(0.75*w), int(0.40*h), w, int(0.70*h)),  # Middle of right pane      
            (int(0.75*w), int(0.70*h), w, int(0.95*h)),  # Bottom of right pane      

            # Additional regions where text might appear      
#            (int(0.60*w), int(0.15*h), w, int(0.50*h)),  # Wider right area      
#            (int(0.50*w), int(0.80*h), w, h),            # Bottom info area      
#            (0, int(0.85*h), w, h),                      # Full bottom strip      
        ]      

        for i, (x1, y1, x2, y2) in enumerate(filter_regions):      
            # Clamp coordinates      
            x1, y1 = max(0, x1), max(0, y1)      
            x2, y2 = min(w, x2), min(h, y2)      

            if x2 <= x1 or y2 <= y1:      
                continue      

            roi = frame[y1:y2, x1:x2]      

            print(f"[FILTER DEBUG] Checking region {i}: ({x1},{y1}) to ({x2},{y2}) - size: {roi.shape}")      

            # Save debug image      
            _save_dbg(f"filter_region_{i}_{filter_text.replace(' ', '_')}.png", roi)      

            # Try both quick and comprehensive checks      
            if self.quick_text_check(roi, filter_text):      
                print(f"[QUICK FILTER] Found '{filter_text}' in region {i}")      
                return True      

            # If quick fails, try comprehensive on this region      
            if self.comprehensive_text_check(roi, filter_text, f"filter_comprehensive_{i}"):      
                print(f"[COMPREHENSIVE FILTER] Found '{filter_text}' in region {i}")      
                return True      

        print(f"[FILTER DEBUG] '{filter_text}' not found in any region")      
        return False     

    def specialized_filter_ocr(self, roi: np.ndarray, target_text: str) -> bool:      
        """    
        Specialized OCR for filter text with strict matching.    

        Performs highly specialized OCR processing designed specifically for    
        filter text recognition, using strict matching criteria to minimize    
        false positives while maintaining high detection accuracy.    

        Args:    
            roi (np.ndarray): Region of interest containing potential filter text    
            target_text (str): Specific filter text to search for    

        Returns:    
            bool: True if filter text is found with high confidence, False otherwise    

        Note:    
            Uses multiple preprocessing approaches and OCR engines with strict    
            matching strategies including exact substring, exact word, and    
            controlled fuzzy matching for single words only.    
        """    
        if roi.size == 0:      
            return False      

        print(f"[SPECIALIZED FILTER OCR] Looking for '{target_text}'")      

        # Try multiple preprocessing approaches      
        preprocessing_approaches = [      
            ("original", lambda img: img),      
            ("grayscale", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img),      
            ("enhanced", lambda img: self._enhance_contrast(img)),      
            ("threshold", lambda img: self._apply_adaptive_threshold(img)),      
        ]      

        # Try multiple OCR configurations      
        ocr_configs = [      
            "--oem 3 --psm 6",  # Default      
            "--oem 3 --psm 7",  # Single line      
            "--oem 3 --psm 8",  # Single word      
            "--oem 3 --psm 13", # Raw line      
            "--oem 1 --psm 6",  # Different OCR engine      
        ]      

        target_upper = target_text.upper()      
        target_words = target_upper.split()      

        for prep_name, prep_func in preprocessing_approaches:      
            try:      
                processed_img = prep_func(roi)      
                _save_dbg(f"specialized_filter_{prep_name}.png", processed_img)      

                for config in ocr_configs:      
                    try:      
                        extracted_text = pytesseract.image_to_string(processed_img, config=config).strip()      

                        if extracted_text:      
                            extracted_upper = extracted_text.upper()      
                            print(f"[SPECIALIZED FILTER OCR] {prep_name}+{config}: '{extracted_text}'")      

                            # STRICT matching strategies - much more discriminating      
                            match_found = False      

                            # Strategy 1: Exact substring match (most reliable)      
                            if target_upper in extracted_upper:      
                                print(f"[SPECIALIZED FILTER OCR] ✓ EXACT MATCH: '{target_text}' found as substring")      
                                match_found = True      

                            # Strategy 2: Exact word match      
                            elif any(target_upper == word for word in extracted_upper.split()):      
                                print(f"[SPECIALIZED FILTER OCR] ✓ EXACT WORD MATCH: '{target_text}' found as complete word")      
                                match_found = True      

                            # Strategy 3: All target words found as complete words      
                            elif len(target_words) > 1:      
                                extracted_words = extracted_upper.split()      
                                if all(word in extracted_words for word in target_words):      
                                    print(f"[SPECIALIZED FILTER OCR] ✓ ALL WORDS MATCH: All words of '{target_text}' found")      
                                    match_found = True      

                            # Strategy 4: Very strict fuzzy match (only for single words with OCR errors)      
                            elif len(target_words) == 1 and len(target_words[0]) >= 4:      
                                extracted_words = extracted_upper.split()      
                                for word in extracted_words:      
                                    if self._strict_fuzzy_match(target_words[0], word):      
                                        print(f"[SPECIALIZED FILTER OCR] ✓ STRICT FUZZY MATCH: '{target_words[0]}' matches '{word}'")      
                                        match_found = True      
                                        break      

                            if match_found:      
                                return True      
                            else:      
                                print(f"[SPECIALIZED FILTER OCR] ✗ NO MATCH: '{target_text}' not found in '{extracted_text}'")      

                    except Exception as e:      
                        print(f"[SPECIALIZED FILTER OCR] {prep_name}+{config} failed: {e}")      
                        continue      

            except Exception as e:      
                print(f"[SPECIALIZED FILTER OCR] Preprocessing {prep_name} failed: {e}")      
                continue      

        return False    

    def _apply_adaptive_threshold(self, img: np.ndarray) -> np.ndarray:      
        """    
        Apply adaptive thresholding for better text extraction.    

        Applies adaptive thresholding techniques that automatically adjust    
        to local image characteristics, improving text extraction in images    
        with varying lighting or contrast conditions.    

        Args:    
            img (np.ndarray): Input image to threshold (BGR or grayscale)    

        Returns:    
            np.ndarray: Binary image with adaptive thresholding applied    

        Note:    
            Uses Gaussian-weighted adaptive thresholding which works well    
            for text extraction in varying lighting conditions.    
        """    
        if len(img.shape) == 3:      
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
        else:      
            gray = img.copy()      

        # Try adaptive threshold      
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                       cv2.THRESH_BINARY, 11, 2)      
        return adaptive      

    def _fuzzy_match(self, target: str, extracted: str, threshold: float = 0.8) -> bool:    
        """    
        Strict fuzzy matching for OCR errors - much more discriminating.    

        Performs fuzzy string matching with strict criteria to handle OCR errors    
        while minimizing false positives. Uses character overlap analysis with    
        position awareness and length validation.    

        Args:    
            target (str): Target text to match against    
            extracted (str): Text extracted from OCR processing    
            threshold (float): Minimum similarity ratio required (default: 0.8)    

        Returns:    
            bool: True if strings match within the specified threshold    

        Note:    
            Applies stricter thresholds for shorter words and includes additional    
            validation for word length similarity to prevent false matches.    
        """    
        target_clean = target.replace(' ', '').upper()    
        extracted_clean = extracted.replace(' ', '').upper()    

        if len(target_clean) == 0:    
            return False    

        # For short targets (like "YouTube"), require very high similarity    
        if len(target_clean) <= 10:    
            threshold = max(threshold, 0.9)  # At least 90% match for short words    

        # Simple character overlap - but must be in similar positions    
        matches = 0    
        for i, char in enumerate(target_clean):    
            # Look for character in nearby positions in extracted text    
            search_start = max(0, i - 1)    
            search_end = min(len(extracted_clean), i + 2)    
            if char in extracted_clean[search_start:search_end]:    
                matches += 1    

        ratio = matches / len(target_clean)    

        # Additional check: target should not be much shorter than a word in extracted    
        extracted_words = extracted.upper().split()    
        target_upper = target.upper()    

        # If target is a single word, it shouldn't match if no word in extracted is similar length    
        if ' ' not in target.strip():    
            similar_length_word_exists = any(    
                abs(len(word) - len(target_upper)) <= 2 for word in extracted_words    
            )    
            if not similar_length_word_exists and ratio < 0.95:    
                return False    

        return ratio >= threshold    

    def _strict_fuzzy_match(self, target: str, candidate: str, min_ratio: float = 0.85) -> bool:    
        """    
        Very strict fuzzy matching for single words only.    

        Implements highly restrictive fuzzy matching designed specifically for    
        single-word comparisons where high precision is required to avoid    
        false positive matches in OCR results.    

        Args:    
            target (str): Target word to match    
            candidate (str): Candidate word from OCR results    
            min_ratio (float): Minimum character match ratio required (default: 0.85)    

        Returns:    
            bool: True if words match within strict criteria, False otherwise    

        Note:    
            Requires similar length (within 2 characters), same starting character,    
            and high character-by-character similarity. Designed to minimize false    
            positives while handling common OCR errors.    
        """    
        if not target or not candidate:    
            return False    

        # Must be similar length (within 2 characters)    
        if abs(len(target) - len(candidate)) > 2:    
            return False    

        # Must have similar starting character    
        if target[0] != candidate[0]:    
            return False    

        # Character-by-character similarity    
        matches = sum(1 for i, char in enumerate(target)    
                     if i < len(candidate) and char == candidate[i])    

        ratio = matches / max(len(target), len(candidate))    

        return ratio >= min_ratio    

    def enhanced_filter_check(self, frame: np.ndarray, filter_text: str) -> bool:    
        """    
        Enhanced filter check with multiple strategies.    

        Performs comprehensive filter text detection using multiple search regions    
        and specialized OCR processing to maximize detection accuracy for content    
        filter information in DVR interfaces.    

        Args:    
            frame (np.ndarray): Full interface frame to search within    
            filter_text (str): Filter text to search for (e.g., genre, rating)    

        Returns:    
            bool: True if filter text is found in any search region, False otherwise    

        Note:    
            Searches multiple regions where filter information typically appears    
            and uses specialized OCR processing optimized for filter text detection.    
        """    
        h, w = frame.shape[:2]    

        print(f"[ENHANCED FILTER] Searching for '{filter_text}' in frame {w}x{h}")    

        # Define search regions (expanded from original)    
        search_regions = [    
            # Original right pane regions    
            ("right_top", int(0.75*w), int(0.20*h), w, int(0.45*h)),    
            ("right_middle", int(0.75*w), int(0.40*h), w, int(0.70*h)),    
            ("right_bottom", int(0.75*w), int(0.70*h), w, int(0.95*h)),    

            # Additional regions    
#            ("wide_right", int(0.60*w), int(0.15*h), w, int(0.80*h)),    
#            ("bottom_strip", 0, int(0.85*h), w, h),    
#            ("center_right", int(0.50*w), int(0.30*h), w, int(0.70*h)),    
        ]    

        for region_name, x1, y1, x2, y2 in search_regions:    
            # Clamp coordinates    
            x1, y1 = max(0, x1), max(0, y1)    
            x2, y2 = min(w, x2), min(h, y2)    

            if x2 <= x1 or y2 <= y1:    
                continue    

            roi = frame[y1:y2, x1:x2]    
            print(f"[ENHANCED FILTER] Checking {region_name}: ({x1},{y1})-({x2},{y2})")    

            # Save debug image    
            _save_dbg(f"enhanced_filter_{region_name}.png", roi)    

            # Try specialized OCR    
            if self.specialized_filter_ocr(roi, filter_text):    
                print(f"[ENHANCED FILTER] ✓ Found '{filter_text}' in {region_name}")    
                return True    

        print(f"[ENHANCED FILTER] ✗ '{filter_text}' not found in any region")    
        return False    

# ─────────────────────────────────────────────────────────────────────────────      
# OPTIMIZED GRID NAVIGATOR      
# ─────────────────────────────────────────────────────────────────────────────      
class GridNavigator:      
    """    
    Enhanced grid navigation with comprehensive recovery including DOWN movement.    

    This class manages intelligent navigation through tile grids in DVR interfaces,    
    providing robust recovery mechanisms when navigation goes outside expected    
    boundaries and comprehensive baseline establishment for consistent operation.    
    """    

    def __init__(self, controller: DeviceController, config: NavigationConfig):      
        """    
        Initialize grid navigator with device controller and configuration.    

        Sets up the navigator with references to the device controller for sending    
        commands and configuration parameters for navigation behavior.    

        Args:    
            controller (DeviceController): Device controller for sending navigation commands    
            config (NavigationConfig): Configuration parameters for navigation behavior    
        """    
        self.controller = controller      
        self.config = config      
        self.initial_highlight_position = None      
        self.grid_bounds = None      
        self.recovery_attempts = 0      
        self.max_recovery_attempts = 3      

    def navigate_to_target(self, mask: int, target_finder: Callable[[np.ndarray], bool],      
                          max_passes: int = None) -> None:      
        """    
        Navigate with enhanced recovery including DOWN movement.    

        Performs systematic navigation through the tile grid, checking each position    
        for the target tile using the provided target finder function. Includes    
        comprehensive recovery mechanisms for handling navigation errors.    

        Args:    
            mask (int): Port bitmask identifying target device    
            target_finder (Callable[[np.ndarray], bool]): Function to test if current tile is target    
            max_passes (int, optional): Maximum number of complete grid traversals    

        Raises:    
            AssertionError: If target is not found after maximum passes or baseline fails    

        Note:    
            Establishes navigation baseline before starting, performs systematic grid    
            traversal with recovery, and provides detailed logging of navigation progress.    
        """    
        max_passes = max_passes or self.config.max_passes      
        max_frame_retries = 3      

        print(f"[NAV] Starting navigation")

        # Establish baseline      
        initial_state = self._establish_baseline(mask, max_frame_retries)      
        if not initial_state:      
            raise AssertionError("Failed to establish navigation baseline")      

        grid_cols, grid_rows = initial_state['cols'], initial_state['rows']      
        print(f"[NAV] Grid dimensions: {grid_cols}x{grid_rows}")      

        # Main navigation loop      
        for nav_pass in range(max_passes):      
            print(f"\n[NAV] === PASS {nav_pass + 1}/{max_passes} ===")      

            if nav_pass > 0:      
                self._comprehensive_recovery_reset(mask, initial_state)      
                time.sleep(0.5)      

            # Traverse the grid      
            for row in range(grid_rows):      
                for col in range(grid_cols):      
                    position_id = f"({row},{col})"      
                    print(f"[NAV] Checking position {position_id}")      

                    # Capture frame with recovery      
                    frame = self._capture_frame_with_recovery(mask, max_frame_retries, initial_state)      
                    if frame is None:      
                        print(f"{YELLOW}[WARN]{RESET} Failed to capture valid frame at {position_id} - skipping")      
                        self._move_to_next_position(mask, row, col, grid_cols, grid_rows)      
                        continue      

                    # Test if this is our target      
                    try:      
                        if target_finder(frame):      
                            print(f"{GREEN}[FOUND]{RESET} TARGET FOUND at pass={nav_pass + 1}, position={position_id}")      
                            return      
                        else:      
                            print(f"[NOT FOUND] Not target at {position_id}")      
                    except Exception as e:      
                        print(f"{RED}[ERROR]{RESET} Target finder error at {position_id}: {e}")      

                    # Move to next position      
                    self._move_to_next_position(mask, row, col, grid_cols, grid_rows)      

            print(f"[NAV] === END OF PASS {nav_pass + 1} ===")      

        raise AssertionError(f"Target not found after {max_passes} passes")      

    def _capture_frame_with_recovery(self, mask: int, max_retries: int, initial_state: dict) -> np.ndarray:      
        """    
        Capture frame with automatic recovery if outside grid area.    

        Attempts to capture a valid frame and automatically recovers navigation    
        if the cursor has moved outside the expected grid area.    

        Args:    
            mask (int): Port bitmask identifying target device    
            max_retries (int): Maximum number of capture attempts    
            initial_state (dict): Baseline navigation state for validation    

        Returns:    
            np.ndarray: Captured frame, or None if all attempts failed    

        Note:    
            Validates captured frames against grid area boundaries and attempts    
            automatic recovery if cursor is detected outside expected regions.    
        """    
        for attempt in range(max_retries):      
            try:      
                time.sleep(self.config.capture_delay)      
                frame = self.controller.grab_current_frame(mask)      

                if frame is not None and frame.size > 0:      
                    # Check if we're in the grid area      
                    if self._verify_in_grid_area(frame, initial_state):      
                        return frame      
                    else:      
                        print(f"[NAV] 🔄 Detected outside grid area - attempting recovery")      
                        if self._attempt_recovery_navigation(mask, initial_state):      
                            # Try capturing again after recovery      
                            time.sleep(0.3)      
                            recovered_frame = self.controller.grab_current_frame(mask)      
                            if recovered_frame is not None and self._verify_in_grid_area(recovered_frame, initial_state):      
                                print(f"[NAV] ✅ Recovery successful")      
                                return recovered_frame      

                        print(f"[NAV] ⚠️  Recovery failed, frame still outside grid area")      
                        return frame  # Return anyway, let caller decide      
                else:      
                    print(f"[NAV] Frame capture attempt {attempt + 1}: Invalid frame")      

            except Exception as e:      
                print(f"[NAV] Frame capture attempt {attempt + 1} failed: {e}")      

            if attempt < max_retries - 1:      
                time.sleep(0.5)      

        return None      

    def _attempt_recovery_navigation(self, mask: int, initial_state: dict) -> bool:      
        """    
        Attempt to recover navigation back to grid area using multiple strategies.    

        Implements multiple recovery strategies to return navigation cursor to the    
        expected grid area when it has moved outside boundaries due to interface    
        changes or navigation errors.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for reference    

        Returns:    
            bool: True if recovery was successful, False otherwise    

        Note:    
            Tries multiple directional recovery strategies (DOWN, RIGHT, LEFT, UP)    
            before falling back to comprehensive reset. Tracks recovery attempts    
            to prevent infinite loops.    
        """    
        self.recovery_attempts += 1      

        if self.recovery_attempts > self.max_recovery_attempts:      
            print(f"[NAV] ⚠️  Max recovery attempts ({self.max_recovery_attempts}) reached")      
            self.recovery_attempts = 0  # Reset for next time      
            return False      

        print(f"[NAV] 🔄 Recovery attempt {self.recovery_attempts}/{self.max_recovery_attempts}")      

        # Strategy 1: Try moving DOWN (we might be in a menu above the grid)      
        if self._try_recovery_down(mask, initial_state):      
            return True      

        # Strategy 2: Try moving RIGHT (we might be in a left sidebar)      
        if self._try_recovery_right(mask, initial_state):      
            return True      

        # Strategy 3: Try moving LEFT (we might have gone too far right)      
        if self._try_recovery_left(mask, initial_state):      
            return True      

        # Strategy 4: Try moving UP (we might have gone below the grid)      
        if self._try_recovery_up(mask, initial_state):      
            return True      

        # Strategy 5: Comprehensive reset      
        print(f"[NAV] 🔄 Trying comprehensive recovery reset")      
        self._comprehensive_recovery_reset(mask, initial_state)      

        # Verify comprehensive reset worked      
        time.sleep(0.5)      
        frame = self.controller.grab_current_frame(mask)      
        if frame is not None and self._verify_in_grid_area(frame, initial_state):      
            print(f"[NAV] ✅ Comprehensive reset successful")      
            return True      

        return False      

    def _try_recovery_down(self, mask: int, initial_state: dict) -> bool:      
        """    
        Try recovery by moving DOWN (common case: we're in upper menu).    

        Attempts to recover navigation by moving down, which handles the common    
        case where the cursor has moved into a menu area above the tile grid.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for validation    

        Returns:    
            bool: True if DOWN recovery was successful, False otherwise    

        Note:    
            Tries up to 3 DOWN movements, checking after each one to see if    
            the cursor has returned to the grid area.    
        """    
        print(f"[NAV] 🔄 Trying recovery: DOWN movement")      

        # Try moving down up to 3 times      
        for i in range(3):      
            self.controller.send_command(mask, "DOWN")      
            time.sleep(0.2)      

            frame = self.controller.grab_current_frame(mask)      
            if frame is not None and self._verify_in_grid_area(frame, initial_state):      
                print(f"[NAV] ✅ DOWN recovery successful after {i+1} moves")      
                return True      

        print(f"[NAV] ❌ DOWN recovery failed")      
        return False      

    def _try_recovery_right(self, mask: int, initial_state: dict) -> bool:      
        """    
        Try recovery by moving RIGHT (case: we're in left sidebar).    

        Attempts to recover navigation by moving right, which handles cases    
        where the cursor has moved into a left sidebar or menu area.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for validation    

        Returns:    
            bool: True if RIGHT recovery was successful, False otherwise    

        Note:    
            Tries up to 2 RIGHT movements, checking after each one to see if    
            the cursor has returned to the grid area.    
        """    
        print(f"[NAV] 🔄 Trying recovery: RIGHT movement")      

        for i in range(2):      
            self.controller.send_command(mask, "RIGHT")      
            time.sleep(0.2)      

            frame = self.controller.grab_current_frame(mask)      
            if frame is not None and self._verify_in_grid_area(frame, initial_state):      
                print(f"[NAV] ✅ RIGHT recovery successful after {i+1} moves")      
                return True      

        print(f"[NAV] ❌ RIGHT recovery failed")      
        return False      

    def _try_recovery_left(self, mask: int, initial_state: dict) -> bool:      
        """    
        Try recovery by moving LEFT (case: we went too far right).    

        Attempts to recover navigation by moving left, which handles cases    
        where the cursor has moved too far to the right of the tile grid.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for validation    

        Returns:    
            bool: True if LEFT recovery was successful, False otherwise    

        Note:    
            Tries up to 3 LEFT movements, checking after each one to see if    
            the cursor has returned to the grid area.    
        """    
        print(f"[NAV] 🔄 Trying recovery: LEFT movement")      

        for i in range(3):      
            self.controller.send_command(mask, "LEFT")      
            time.sleep(0.2)      

            frame = self.controller.grab_current_frame(mask)      
            if frame is not None and self._verify_in_grid_area(frame, initial_state):      
                print(f"[NAV] ✅ LEFT recovery successful after {i+1} moves")      
                return True      

        print(f"[NAV] ❌ LEFT recovery failed")      
        return False      

    def _try_recovery_up(self, mask: int, initial_state: dict) -> bool:      
        """    
        Try recovery by moving UP (case: we went below the grid).    

        Attempts to recover navigation by moving up, which handles cases    
        where the cursor has moved below the tile grid area.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for validation    

        Returns:    
            bool: True if UP recovery was successful, False otherwise    

        Note:    
            Tries up to 2 UP movements, checking after each one to see if    
            the cursor has returned to the grid area.    
        """    
        print(f"[NAV] 🔄 Trying recovery: UP movement")      

        for i in range(2):      
            self.controller.send_command(mask, "UP")      
            time.sleep(0.2)      

            frame = self.controller.grab_current_frame(mask)      
            if frame is not None and self._verify_in_grid_area(frame, initial_state):      
                print(f"[NAV] ✅ UP recovery successful after {i+1} moves")      
                return True      

        print(f"[NAV] ❌ UP recovery failed")      
        return False      

    def _comprehensive_recovery_reset(self, mask: int, initial_state: dict):      
        """    
        Comprehensive recovery reset with multiple strategies.    

        Performs a comprehensive reset of navigation position using multiple    
        strategies to return to a known good state within the tile grid.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for reference    

        Note:    
            Implements a multi-step recovery process: DOWN-first recovery to handle    
            upper menus, limited left-up reset, and additional DOWN attempts if needed.    
            Designed to handle various interface layouts and navigation scenarios.    
        """    
        print(f"[NAV] 🔄 Performing comprehensive recovery reset")      

        # Strategy 1: Try to get back to grid area by moving DOWN first      
        # (This handles the common case where we're in an upper menu)      
        print(f"[NAV] 🔄 Step 1: Attempting DOWN-first recovery")      
        for i in range(4):  # Try moving down up to 4 times      
            frame_before = self.controller.grab_current_frame(mask)      
            self.controller.send_command(mask, "DOWN")      
            time.sleep(0.15)      

            frame_after = self.controller.grab_current_frame(mask)      

            # Check if we found the grid area      
            if frame_after is not None and self._verify_in_grid_area(frame_after, initial_state):      
                print(f"[NAV] ✅ Found grid area after {i+1} DOWN moves")      
                # Now try to get to top-left of grid      
                self._safe_reset_within_grid(mask, initial_state)      
                return      

            # Check if frame stopped changing (hit bottom boundary)      
            if (frame_before is not None and frame_after is not None and       
                self._frames_similar(frame_before, frame_after, threshold=0.95)):      
                print(f"[NAV] 🔄 Hit boundary after {i+1} DOWN moves, stopping")      
                break      

        # Strategy 2: Traditional left-up reset (but limited)      
        print(f"[NAV] 🔄 Step 2: Limited left-up reset")      
        self._safe_reset_to_grid_area(mask, initial_state)      

        # Strategy 3: If still not in grid, try DOWN again      
        frame = self.controller.grab_current_frame(mask)      
        if frame is not None and not self._verify_in_grid_area(frame, initial_state):      
            print(f"[NAV] 🔄 Step 3: Additional DOWN attempts")      
            for i in range(3):      
                self.controller.send_command(mask, "DOWN")      
                time.sleep(0.2)      
                frame = self.controller.grab_current_frame(mask)      
                if frame is not None and self._verify_in_grid_area(frame, initial_state):      
                    print(f"[NAV] ✅ Found grid area with additional DOWN")      
                    self._safe_reset_within_grid(mask, initial_state)      
                    return      

    def _safe_reset_within_grid(self, mask: int, initial_state: dict):      
        """    
        Reset to top-left within the grid area (assumes we're already in grid).    

        Performs a safe reset to the top-left position of the grid while ensuring    
        the cursor remains within the grid boundaries throughout the process.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for boundary validation    

        Note:    
            Uses conservative movement with boundary checking to prevent moving    
            outside the grid area during the reset process. Stops movement if    
            frame changes indicate leaving the grid area.    
        """    
        print(f"[NAV] 🔄 Resetting to top-left within grid area")      

        # Move left conservatively      
        for i in range(initial_state['cols'] + 1):      
            frame_before = self.controller.grab_current_frame(mask)      
            self.controller.send_command(mask, "LEFT")      
            time.sleep(0.05)      

            frame_after = self.controller.grab_current_frame(mask)      

            # Stop if we're about to leave grid area      
            if (frame_after is not None and       
                not self._verify_in_grid_area(frame_after, initial_state)):      
                print(f"[NAV] 🔄 Stopping LEFT movement to stay in grid")      
                self.controller.send_command(mask, "RIGHT")  # Move back      
                time.sleep(0.05)      
                break      

            # Stop if no change (hit boundary)      
            if (frame_before is not None and frame_after is not None and       
                self._frames_similar(frame_before, frame_after, threshold=0.95)):      
                break      

        # Move up conservatively      
        for i in range(initial_state['rows'] + 1):      
            frame_before = self.controller.grab_current_frame(mask)      
            self.controller.send_command(mask, "UP")      
            time.sleep(0.05)      

            frame_after = self.controller.grab_current_frame(mask)      

            # Stop if we're about to leave grid area      
            if (frame_after is not None and       
                not self._verify_in_grid_area(frame_after, initial_state)):      
                print(f"[NAV] 🔄 Stopping UP movement to stay in grid")      
                self.controller.send_command(mask, "DOWN")  # Move back      
                time.sleep(0.05)      
                break      

            # Stop if no change (hit boundary)      
            if (frame_before is not None and frame_after is not None and       
                self._frames_similar(frame_before, frame_after, threshold=0.95)):      
                break      

    def _establish_baseline(self, mask: int, max_retries: int) -> dict:    
        """    
        Establish navigation baseline without moving cursor.    

        Creates a baseline understanding of the current navigation state including    
        grid dimensions, highlight characteristics, and boundary definitions without    
        moving the cursor from its current position.    

        Args:    
            mask (int): Port bitmask identifying target device    
            max_retries (int): Maximum number of frame capture attempts    

        Returns:    
            dict: Baseline state information including grid dimensions and boundaries,    
                 or None if baseline establishment failed    

        Note:    
            Analyzes current highlight rectangle to determine grid characteristics    
            and establishes approximate boundaries for grid area validation.    
        """    
        frame = self._capture_frame_with_retries(mask, max_retries)    
        if frame is None:    
            return None    

        # Initialize highlight detector    
        highlight_detector = HighlightDetector(ImageProcessor(VisionConfig()))    
        hi_rect = highlight_detector.find_highlight_rect_strict(frame, debug_prefix="baseline")    

        if hi_rect is None:    
            return None    

        x, y, w, h = hi_rect    

        # Store initial highlight position and characteristics    
        self.initial_highlight_position = (x, y, w, h)    

        # Determine grid based on tile characteristics    
        grid_info = self._analyze_grid_from_highlight(frame, hi_rect)    

        # Establish grid bounds (approximate area where tiles should be)    
        self.grid_bounds = {    
            'left': max(0, x - w),  # Allow some margin    
            'right': min(frame.shape[1], x + (grid_info['cols'] * w) + w),    
            'top': max(0, y - h//2),  # Small margin above    
            'bottom': min(frame.shape[0], y + (grid_info['rows'] * h) + h//2)    
        }    

        return {    
            'cols': grid_info['cols'],    
            'rows': grid_info['rows'],    
            'position': (0, 0),  # Assume we start at 0,0    
            'highlight_rect': hi_rect,    
            'frame_signature': self._get_frame_signature(frame),    
            'grid_bounds': self.grid_bounds    
        }    

    def _analyze_grid_from_highlight(self, frame: np.ndarray, hi_rect: tuple) -> dict:    
        """    
        Analyze grid dimensions from current highlight without moving.    

        Determines the likely grid dimensions based on the size and position    
        of the current highlight rectangle, using heuristics based on tile    
        sizes commonly found in DVR interfaces.    

        Args:    
            frame (np.ndarray): Current interface frame    
            hi_rect (tuple): Highlight rectangle (x, y, width, height)    

        Returns:    
            dict: Grid information containing 'cols' and 'rows' estimates    

        Note:    
            Uses tile width to estimate column count and available vertical    
            space to estimate row count. Based on common DVR interface layouts.    
        """    
        x, y, w, h = hi_rect    

        # Determine columns based on tile size    
        if w > 150:  # Large tiles    
            cols = 4    
        elif w > 120:  # Medium-large tiles    
            cols = 5    
        elif w > 90:   # Medium tiles    
            cols = 6    
        else:  # Small tiles    
            cols = 7    

        # Determine rows based on position and available space    
        frame_height = frame.shape[0]    
        available_height = frame_height - y - h    

        if available_height > h * 1.5:  # Room for more rows    
            rows = 2    
        elif available_height > h * 2.5:  # Room for even more rows    
            rows = 3    
        else:    
            rows = 1    

        return {'cols': cols, 'rows': rows}    

    def _safe_reset_to_grid_area(self, mask: int, initial_state: dict):    
        """    
        Safe reset that doesn't go above the grid area.    

        Performs a conservative reset to the grid area that avoids moving    
        into menu areas above the tile grid, using limited movements and    
        boundary checking.    

        Args:    
            mask (int): Port bitmask identifying target device    
            initial_state (dict): Baseline navigation state for validation    

        Note:    
            Uses conservative movement limits and stops if significant frame    
            changes are detected, indicating movement into different interface areas.    
        """    
        print(f"[NAV] Performing safe reset to grid area")    

        # Instead of going to absolute top-left, move conservatively    
        # Move left until we can't go further (but limit movements)    
        max_left_moves = initial_state['cols'] + 1    
        for i in range(max_left_moves):    
            self.controller.send_command(mask, "LEFT")    
            time.sleep(0.05)    

            # Check if we've moved outside expected area    
            frame = self.controller.grab_current_frame(mask)    
            if frame is not None and not self._verify_in_grid_area(frame, initial_state):    
                print(f"[NAV] Detected movement outside grid - stopping left movement")    
                # Move back right once    
                self.controller.send_command(mask, "RIGHT")    
                time.sleep(0.05)    
                break    

        # Move up conservatively (limited movements)    
        max_up_moves = initial_state['rows'] + 1    
        for i in range(max_up_moves):    
            frame_before = self.controller.grab_current_frame(mask)    
            self.controller.send_command(mask, "UP")    
            time.sleep(0.05)    

            frame_after = self.controller.grab_current_frame(mask)    

            # If frame changed significantly, we might have moved to a different menu      
            if (frame_before is not None and frame_after is not None and      
                not self._frames_similar(frame_before, frame_after)):      
                print(f"[NAV] Detected significant frame change - stopping up movement")      
                # Move back down once      
                self.controller.send_command(mask, "DOWN")      
                time.sleep(0.05)      
                break      

            # Also check if we're still in grid area      
            if frame_after is not None and not self._verify_in_grid_area(frame_after, initial_state):      
                print(f"[NAV] Moved outside grid area - stopping up movement")      
                self.controller.send_command(mask, "DOWN")      
                time.sleep(0.05)      
                break      

        time.sleep(0.3)  # Allow UI to settle      

    def _verify_in_grid_area(self, frame: np.ndarray, initial_state: dict) -> bool:      
        """      
        Verify that we're still in the expected grid area.      

        Validates that the current navigation position is still within the      
        expected tile grid area by checking highlight position and size      
        against established boundaries.      

        Args:      
            frame (np.ndarray): Current interface frame to validate      
            initial_state (dict): Baseline navigation state with grid boundaries      

        Returns:      
            bool: True if cursor is within expected grid area, False otherwise      

        Note:      
            Uses highlight rectangle detection and compares position and size      
            against established grid boundaries and initial highlight characteristics.      
        """      
        if frame is None or self.grid_bounds is None:      
            return True  # Can't verify, assume OK      

        # Look for highlight rectangle      
        highlight_detector = HighlightDetector(ImageProcessor(VisionConfig()))      
        hi_rect = highlight_detector.find_highlight_rect_strict(frame)      

        if hi_rect is None:      
            return False  # No highlight found      

        x, y, w, h = hi_rect      

        # Check if highlight is within expected grid bounds      
        in_bounds = (      
            self.grid_bounds['left'] <= x <= self.grid_bounds['right'] and      
            self.grid_bounds['top'] <= y <= self.grid_bounds['bottom']      
        )      

        # Also check if highlight size is similar to initial      
        initial_w, initial_h = initial_state['highlight_rect'][2], initial_state['highlight_rect'][3]      
        size_similar = (      
            abs(w - initial_w) <= initial_w * 0.3 and  # Within 30% of original size      
            abs(h - initial_h) <= initial_h * 0.3      
        )      

        return in_bounds and size_similar      

    def _frames_similar(self, frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.85) -> bool:      
        """      
        Check if two frames are similar (to detect menu changes).      

        Compares two frames to determine if they represent similar interface      
        states, used to detect when navigation has moved into different      
        menu areas or interface sections.      

        Args:      
            frame1 (np.ndarray): First frame for comparison      
            frame2 (np.ndarray): Second frame for comparison      
            threshold (float): Similarity threshold (default: 0.85)      

        Returns:      
            bool: True if frames are similar above threshold, False otherwise      

        Note:      
            Uses histogram comparison for efficient similarity detection.      
            Returns True if comparison fails to avoid false navigation errors.      
        """      
        if frame1 is None or frame2 is None:      
            return False      

        if frame1.shape != frame2.shape:      
            return False      

        # Simple similarity check using histogram comparison      
        try:      
            hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])      
            hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])      

            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)      
            return correlation >= threshold      
        except:      
            return True  # If comparison fails, assume similar      

    def _get_frame_signature(self, frame: np.ndarray) -> str:      
        """      
        Get a simple signature of the frame for comparison.      

        Generates a simple hash-based signature of the frame for quick      
        comparison and change detection during navigation operations.      

        Args:      
            frame (np.ndarray): Frame to generate signature for      

        Returns:      
            str: Hash-based signature string, or empty string on error      

        Note:      
            Uses downsampled frame hash for efficient comparison while      
            maintaining reasonable sensitivity to significant changes.      
        """      
        if frame is None:      
            return ""      
        try:      
            # Simple hash of downsampled frame      
            small = cv2.resize(frame, (32, 32))      
            return str(hash(small.tobytes()))      
        except:      
            return ""      

    def _move_to_next_position(self, mask: int, current_row: int, current_col: int,      
                              grid_cols: int, grid_rows: int):      
        """      
        Move to next position with boundary checking.      

        Handles systematic movement to the next position in the grid,      
        managing row transitions and boundary conditions appropriately.      

        Args:      
            mask (int): Port bitmask identifying target device      
            current_row (int): Current row position in grid      
            current_col (int): Current column position in grid      
            grid_cols (int): Total number of columns in grid      
            grid_rows (int): Total number of rows in grid      

        Note:      
            Moves right within rows, then moves to start of next row when      
            reaching end of current row. Handles grid boundary conditions.      
        """      
        if current_col < grid_cols - 1:      
            print(f"[NAV] Moving RIGHT from ({current_row},{current_col})")      
            self.controller.send_command(mask, "RIGHT")      
            time.sleep(self.config.move_delay)      
        elif current_row < grid_rows - 1:      
            print(f"[NAV] End of row {current_row}, moving to start of row {current_row + 1}")      
            # Move to leftmost position of next row      
            for _ in range(current_col):      
                self.controller.send_command(mask, "LEFT")      
                time.sleep(0.05)      
            self.controller.send_command(mask, "DOWN")      
            time.sleep(self.config.move_delay)      

    def _capture_frame_with_retries(self, mask: int, max_retries: int) -> np.ndarray:      
        """      
        Capture frame with retries and validation.      

        Attempts to capture a valid frame with multiple retry attempts,      
        including validation to ensure the captured frame is usable.      

        Args:      
            mask (int): Port bitmask identifying target device      
            max_retries (int): Maximum number of capture attempts      

        Returns:      
            np.ndarray: Valid captured frame, or None if all attempts failed      

        Note:      
            Includes delays between attempts and validates frame size and      
            content before returning. Used during baseline establishment.      
        """      
        for attempt in range(max_retries):      
            try:      
                time.sleep(self.config.capture_delay)      
                frame = self.controller.grab_current_frame(mask)      

                if frame is not None and frame.size > 0:      
                    return frame      
                else:      
                    print(f"[NAV] Frame capture attempt {attempt + 1}: Invalid frame")      

            except Exception as e:      
                print(f"[NAV] Frame capture attempt {attempt + 1} failed: {e}")      

            if attempt < max_retries - 1:      
                time.sleep(0.5)      

        return None      

# ─────────────────────────────────────────────────────────────────────────────      
# OPTIMIZED GENERALIZED TILE FINDER      
# ─────────────────────────────────────────────────────────────────────────────      
class GeneralizedTileFinder:      
    """      
    Optimized tile finder with smart sequential processing.      

    This is the main class that orchestrates the tile finding process, combining      
    template matching, OCR text detection, and content filtering to locate and      
    select specific tiles in DVR interfaces. It provides optimized processing      
    with early returns and comprehensive error handling.      
    """      

    def __init__(self, config: AppConfig):      
        """      
        Initialize the tile finder with optimized components.      

        Sets up all the necessary components for tile finding including device      
        controller, image processor, highlight detector, OCR engine, and navigator.      

        Args:      
            config (AppConfig): Complete application configuration      

        Note:      
            Creates instances of all processing components with shared configuration      
            to ensure consistent behavior across all tile finding operations.      
        """      
        self.config = config      
        self.controller = DeviceController(config)      
        self.image_processor = ImageProcessor(config.vision)      
        self.highlight_detector = HighlightDetector(self.image_processor)      
        self.ocr_engine = OCREngine(self.image_processor)      
        self.navigator = GridNavigator(self.controller, config.navigation)      

    def find_tile_with_integrated_filter(self,      
                                       port_mask: int,      
                                       template_name: str = None,      
                                       label_text: str = None,      
                                       template_threshold: float = 0.6,      
                                       require_both: bool = False,      
                                       filter_text: str = None,      
                                       select_when_found: bool = True) -> bool:      
        """      
        Optimized tile finder with smart early returns.      

        Main tile finding method that combines template matching, text detection,      
        and content filtering to locate target tiles. Uses optimized processing      
        with early returns to maximize performance while maintaining accuracy.      

        Args:      
            port_mask (int): Port bitmask identifying target device      
            template_name (str, optional): Template image name (without .png extension)      
            label_text (str, optional): Text to find in tile label      
            template_threshold (float): Template matching confidence threshold (default: 0.6)      
            require_both (bool): Whether both template AND text must match (default: False)      
            filter_text (str, optional): Additional content filter text      
            select_when_found (bool): Whether to select tile when found (default: True)      

        Returns:      
            bool: True if target tile was found (and optionally selected), False otherwise      

        Raises:      
            ValueError: If neither template_name nor label_text is specified      

        Note:      
            Uses smart early return logic to skip unnecessary processing when      
            criteria are already met. Provides comprehensive statistics and      
            debugging information for analysis and troubleshooting.      
        """      

        if not template_name and not label_text:      
            raise ValueError("Must specify either template_name or label_text (or both)")      

        tiles_checked = 0      
        success_stats = {"template": 0, "text": 0, "filter": 0}      

        def is_target_tile_optimized(frame: np.ndarray) -> bool:      
            """      
            Optimized tile checker with early returns.      

            Internal function that performs the actual tile checking logic      
            with optimized processing order and early return conditions.      

            Args:      
                frame (np.ndarray): Current interface frame to check      

            Returns:      
                bool: True if current tile matches all specified criteria      

            Note:      
                Uses nonlocal variables to track statistics and implements      
                smart processing order to minimize unnecessary operations.      
            """      
            nonlocal tiles_checked      
            tiles_checked += 1      

            print(f"\n[OPTIMIZED CHECK #{tiles_checked}] ==================")      

            # Find highlight rectangle      
            hi_rect = self.highlight_detector.find_highlight_rect_strict(frame, debug_prefix=f"tile_{tiles_checked:02d}")      
            if not hi_rect:      
                print(f"[CHECK #{tiles_checked}] No highlight found")      
                return False      

            print(f"[CHECK #{tiles_checked}] Highlight found: {hi_rect}")      

            # Extract tile content      
            tile_content = self._extract_larger_tile_region(frame, hi_rect)      

            # STEP 1: Check template (fast)      
            template_match = False      
            if template_name:      
                template_path = str(self.config.template_dir / f"{template_name}.png")      
                if os.path.exists(template_path):      
                    print(f"[CHECK #{tiles_checked}] Checking template: {template_name}")      
                    template_match = _tile_has_logo_fast(tile_content, template_path, template_threshold)      
                    print(f"[CHECK #{tiles_checked}] Template result: {template_match}")      
                    if template_match:      
                        success_stats["template"] += 1      

            # STEP 2: Check text (potentially slow, so optimize)      
            text_match = False      
            if label_text:      
                # Only check text if we need to (early return optimization)      
                if require_both and not template_match:      
                    print(f"[CHECK #{tiles_checked}] Skipping text check - template required but failed")      
                elif not require_both and template_match:      
                    print(f"[CHECK #{tiles_checked}] Skipping text check - template succeeded and either/or mode")      
                    text_match = True  # Don't need to actually check      
                else:      
                    print(f"[CHECK #{tiles_checked}] Checking text: '{label_text}'")      
                    text_match = self._quick_text_check_around_tile(frame, hi_rect, label_text)      
                    print(f"[CHECK #{tiles_checked}] Text result: {text_match}")      
                    if text_match:      
                        success_stats["text"] += 1      

            # STEP 3: Evaluate basic criteria      
            if require_both:      
                basic_match = template_match and text_match      
            else:      
                basic_match = template_match or text_match      

            print(f"[CHECK #{tiles_checked}] Basic match: {basic_match} (template: {template_match}, text: {text_match})")      

            if not basic_match:      
                print(f"[CHECK #{tiles_checked}] Basic criteria failed - tile rejected")      
                return False      

            # STEP 4: Check filter (only if basic criteria passed)      
            if filter_text:      
                print(f"[CHECK #{tiles_checked}] Checking filter: '{filter_text}'")      
                filter_match = self.ocr_engine.enhanced_filter_check(frame, filter_text)      
                print(f"[CHECK #{tiles_checked}] Filter result: {filter_match}")      

                if not filter_match:      
                    if self.config.debug_enabled:      
                        self.debug_failed_filter(frame, filter_text, tiles_checked)      
                    print(f"[CHECK #{tiles_checked}] Filter check failed - tile rejected")      
                    return False      

                success_stats["filter"] += 1      

            # All checks passed!      
            print(f"[CHECK #{tiles_checked}] ✓ ALL CHECKS PASSED - TARGET FOUND!")      
            print(f"[CHECK #{tiles_checked}] ==================\n")      
            return True      

        try:      
            print(f"\n[OPTIMIZED SEARCH] Starting search...")      
            print(f"[OPTIMIZED SEARCH] Template: {template_name}, Text: {label_text}, Filter: {filter_text}")      
            print(f"[OPTIMIZED SEARCH] Require both: {require_both}, Threshold: {template_threshold}")      

            self.navigator.navigate_to_target(port_mask, is_target_tile_optimized)      

            print(f"\n[OPTIMIZED SEARCH SUMMARY] ==================")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Total tiles checked: {tiles_checked}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Template successes: {success_stats['template']}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Text successes: {success_stats['text']}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Filter successes: {success_stats['filter']}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] ✓ SUCCESS - Target found!")      

            if select_when_found:      
                time.sleep(0.5)      
                self.controller.send_command(port_mask, "SELECT")      
                print(f"[OPTIMIZED SEARCH SUMMARY] Tile selected!")      

            return True      

        except AssertionError as e:      
            print(f"\n[OPTIMIZED SEARCH SUMMARY] ==================")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Total tiles checked: {tiles_checked}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Template successes: {success_stats['template']}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Text successes: {success_stats['text']}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] Filter successes: {success_stats['filter']}")      
            print(f"[OPTIMIZED SEARCH SUMMARY] ✗ FAILED - {e}")      
            return False      

    def _extract_larger_tile_region(self, frame: np.ndarray, hi_rect: Tuple[int, int, int, int]) -> np.ndarray:      
        """      
        Extract a larger tile region to ensure we get the full logo.      

        Extracts an expanded region around the highlight rectangle to ensure      
        complete logo capture, accounting for logos that may extend slightly      
        beyond the highlight boundaries.      

        Args:      
            frame (np.ndarray): Full interface frame      
            hi_rect (Tuple[int, int, int, int]): Highlight rectangle (x, y, width, height)      

        Returns:      
            np.ndarray: Expanded tile region containing the complete logo area      

        Note:      
            Adds padding around the highlight rectangle while respecting      
            frame boundaries to prevent index errors during extraction.      
        """      
        x, y, w, h = hi_rect      

        # Expand the region slightly      
        padding = 10      
        x1 = max(0, x - padding)      
        y1 = max(0, y - padding)      
        x2 = min(frame.shape[1], x + w + padding)      
        y2 = min(frame.shape[0], y + h + padding)      

        return frame[y1:y2, x1:x2]      

    def _quick_text_check_around_tile(self, frame: np.ndarray, hi_rect: Tuple[int, int, int, int], target_text: str) -> bool:      
        """      
        Quick text check in most likely locations around tile.      

        Performs optimized text detection by checking only the most likely      
        locations where tile labels typically appear, prioritizing speed      
        while maintaining good detection accuracy.      

        Args:      
            frame (np.ndarray): Full interface frame      
            hi_rect (Tuple[int, int, int, int]): Highlight rectangle (x, y, width, height)      
            target_text (str): Text to search for in tile labels      

        Returns:      
            bool: True if target text is found in any likely location      

        Note:      
            Checks regions in order of likelihood: below tile, bottom of tile,      
            extended below area. Falls back to comprehensive check if quick      
            methods fail on the most likely region.      
        """      
        x, y, w, h = hi_rect      
        frame_h, frame_w = frame.shape[:2]      

        # Check only the most likely locations (in order of probability)      
        quick_regions = [      
            # Below tile (most common for labels)      
            (x, y + h + int(0.05 * h), x + w, y + h + int(0.6 * h)),      
            # Bottom of tile (sometimes text is inside)      
            (x + 5, y + int(0.7 * h), x + w - 5, y + h - 5),      
            # Extended below (for longer labels)      
            (x - int(0.1 * w), y + h, x + w + int(0.1 * w), y + h + int(0.8 * h)),      
        ]      

        for i, (x1, y1, x2, y2) in enumerate(quick_regions):      
            # Clamp coordinates      
            x1, y1 = max(0, x1), max(0, y1)      
            x2, y2 = min(frame_w, x2), min(frame_h, y2)      

            if x2 <= x1 or y2 <= y1:      
                continue      

            roi = frame[y1:y2, x1:x2]      

            if self.config.debug_enabled:      
                _save_dbg(f"quick_text_region_{i}.png", roi)      

            # Try quick text check first      
            if self.ocr_engine.quick_text_check(roi, target_text):      
                print(f"[QUICK TEXT] Found '{target_text}' in region {i}")      
                return True      

        # If quick checks fail, try comprehensive check on most likely region      
        if quick_regions:      
            x1, y1, x2, y2 = quick_regions[0]  # Below tile region      
            x1, y1 = max(0, x1), max(0, y1)      
            x2, y2 = min(frame_w, x2), min(frame_h, y2)      

            if x2 > x1 and y2 > y1:      
                roi = frame[y1:y2, x1:x2]      
                print(f"[COMPREHENSIVE TEXT] Trying comprehensive check...")      
                return self.ocr_engine.comprehensive_text_check(roi, target_text, "comprehensive_text")      

        return False      

    # Legacy methods for backward compatibility      
    def find_tile_by_logo(self, port_mask: int, logo_name: str, threshold: float = None,      
                         select_when_found: bool = True) -> bool:      
        """      
        Legacy method - find tile by logo only.      

        Backward compatibility method that finds tiles based solely on      
        template/logo matching without text or filter requirements.      

        Args:      
            port_mask (int): Port bitmask identifying target device      
            logo_name (str): Template image name (without .png extension)      
            threshold (float, optional): Template matching threshold (default: 0.6)      
            select_when_found (bool): Whether to select tile when found (default: True)      

        Returns:      
            bool: True if tile with matching logo was found and optionally selected      

        Note:      
            This is a convenience wrapper around the main find_tile_with_integrated_filter      
            method, maintained for backward compatibility with existing code.      
        """      
        return self.find_tile_with_integrated_filter(      
            port_mask=port_mask,      
            template_name=logo_name,      
            template_threshold=threshold or 0.6,      
            select_when_found=select_when_found      
        )      

    def find_tile_by_text(self, port_mask: int, target_text: str,      
                         select_when_found: bool = True) -> bool:      
        """      
        Legacy method - find tile by text only.      

        Backward compatibility method that finds tiles based solely on      
        text label matching without template or filter requirements.      

        Args:      
            port_mask (int): Port bitmask identifying target device      
            target_text (str): Text to find in tile labels      
            select_when_found (bool): Whether to select tile when found (default: True)      

        Returns:      
            bool: True if tile with matching text was found and optionally selected      

        Note:      
            This is a convenience wrapper around the main find_tile_with_integrated_filter      
            method, maintained for backward compatibility with existing code.      
        """      
        return self.find_tile_with_integrated_filter(      
            port_mask=port_mask,      
            label_text=target_text,      
            select_when_found=select_when_found      
        )      

    def debug_failed_filter(self, frame: np.ndarray, filter_text: str, tile_number: int):      
        """      
        Save full frame and regions when filter check fails for debugging.      

        Generates comprehensive debug output when filter checks fail, saving      
        the full frame and all potential filter regions for manual analysis      
        and troubleshooting of filter detection issues.      

        Args:      
            frame (np.ndarray): Full interface frame where filter check failed      
            filter_text (str): Filter text that was being searched for      
            tile_number (int): Sequential number of the tile being checked      

        Note:      
            Saves multiple debug images including full frame and various regions      
            where filter text might appear. Used for improving filter detection      
            accuracy and troubleshooting detection failures.      
        """      
        print(f"[DEBUG FAILED FILTER] Saving debug info for tile #{tile_number}, filter: '{filter_text}'")      

        # Save full frame      
        _save_dbg(f"failed_filter_tile_{tile_number:02d}_full_frame.png", frame)      

        # Save all potential regions      
        h, w = frame.shape[:2]      
        regions = [      
            ("right_pane", int(0.75*w), int(0.20*h), w, int(0.95*h)),      
            ("bottom_area", 0, int(0.80*h), w, h),      
            ("full_right", int(0.50*w), 0, w, h),      
        ]      

        for name, x1, y1, x2, y2 in regions:      
            x1, y1 = max(0, x1), max(0, y1)      
            x2, y2 = min(w, x2), min(h, y2)      
            if x2 > x1 and y2 > y1:      
                roi = frame[y1:y2, x1:x2]      
                _save_dbg(f"failed_filter_tile_{tile_number:02d}_{name}.png", roi)      

