from DAS_library import *
from DAS_library import _tile_has_logo_fast, _save_dbg, execute_classifyfailure
"""
Copyright (c) Dish Technologies
All Rights Reserved
Confidential Property of Dish Technologies

THE MATERIAL CONTAINED HEREIN MAY ONLY BE USED SUBJECT TO AN EXECUTED
AGREEMENT BETWEEN THE USER AND DISH TECHNOLOGIES
DISCLOSURE OF THIS MATERIAL IS STRICTLY PROHIBITED EXCEPT SUBJECT TO
THE TERMS OF SUCH AN AGREEMENT.
"""
# ─────────────────────────────────────────────────────────────────────────────  
# MAIN FUNCTION  
# ─────────────────────────────────────────────────────────────────────────────  
def main():  
    """      
    Main entry point for the generalized tile finder application.      

    Handles command line argument parsing, configuration setup, and execution      
    of the tile finding operation. Provides comprehensive error handling and      
    logging for debugging and analysis purposes.      

    Note:      
        Uses CLIHandler for argument parsing and creates GeneralizedTileFinder      
        instance for actual tile finding operations. Includes error handling      
        for various failure scenarios and calls failure classification on errors.      
    """    

    # Extract UUID from the last argument, ignoring the 3 IDs before it  
    uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'  
    uiStepHistoryId = None  

    # Check if we have enough arguments and if the last one is a UUID  
    if re.match(uuid_pattern, sys.argv[-1]):  
        uiStepHistoryId = sys.argv[-1]  
        # Remove the last 4 arguments (3 IDs + 1 UUID) before argparse processes  
        sys.argv = sys.argv[:-4] 

    # Initialize CLI handler and parse arguments  

    cli_handler = CLIHandler()  

    try:  
        args = cli_handler.parse_args()  
    except SystemExit:  
        # argparse calls sys.exit() on help or error  
        return 1  

    # Validate that at least one search criterion is provided  
    if not args.template and not args.label:  
        print("[ERROR] Must specify either --template (-t) or --label (-l)...or both")  
        return 1  

    # Create configuration from command line arguments  
    config = cli_handler.create_config_from_args(args)  

    # Enable debug mode if requested  
    if args.debug:  
        global show_image  
        show_image = True  
        print(f"[DEBUG] Debug mode enabled. Images will be saved to: {config.debug_dir}")  
        print(f"[DEBUG] Template directory: {config.template_dir}")  

    # Validate template file exists if specified  
    if args.template:  
        template_path = config.template_dir / f"{args.template}.png"  
        if not template_path.exists():  
            print(f"[ERROR] Template file not found: {template_path}")  
            return 1  
        print(f"[INFO] Using template: {template_path}")  

    # Display search parameters  
    print(f"\n[SEARCH CONFIG] ==================")  
    print(f"[SEARCH CONFIG] Port mask: {args.port_mask}")  
    print(f"[SEARCH CONFIG] UiStepHistoryId: {uiStepHistoryId or 'None'}")  
    print(f"[SEARCH CONFIG] Template: {args.template or 'None'}")  
    print(f"[SEARCH CONFIG] Text: {args.label or 'None'}")  
    print(f"[SEARCH CONFIG] Filter: {args.filter_text or 'None'}")  
    print(f"[SEARCH CONFIG] Require both: {args.require_both}")  
    print(f"[SEARCH CONFIG] Template threshold: {args.template_threshold}")  
    print(f"[SEARCH CONFIG] Max passes: {config.navigation.max_passes}")  
    print(f"[SEARCH CONFIG] Grid size: Auto-detected during navigation")
    print(f"[SEARCH CONFIG] ==================\n")  

    # Initialize the tile finder  
    try:  
        tile_finder = GeneralizedTileFinder(config)  
        print(f"[INFO] Tile finder initialized successfully")  
    except Exception as e:  
        print(f"[ERROR] Failed to initialize tile finder: {e}")  
        return 1  

    # Perform the search  
    start_time = time.time()  

    try:  
        print(f"[INFO] Starting tile search...")  

        success = tile_finder.find_tile_with_integrated_filter(  
            port_mask=args.port_mask,  
            template_name=args.template,  
            label_text=args.label,  
            template_threshold=args.template_threshold,  
            require_both=args.require_both,  
            filter_text=args.filter_text,  
            select_when_found=True  # Always select when found  
        )  

        elapsed_time = time.time() - start_time  

        if success:  
            print(f"\n[SUCCESS] ✓ Tile found and selected successfully!")  
            print(f"[SUCCESS] Search completed in {elapsed_time:.2f} seconds")  
            return 0  
        else:  
            if uiStepHistoryId :
                execute_classifyfailure(uiStepHistoryId )
            print(f"\n[FAILURE] ✗ Tile not found")  
            print(f"[FAILURE] Search completed in {elapsed_time:.2f} seconds")  
            return 1  

    except KeyboardInterrupt:  
        print(f"\n[INFO] Search interrupted by user")  
        return 130  # Standard exit code for Ctrl+C  

    except Exception as e:  
        elapsed_time = time.time() - start_time  
        if uiStepHistoryId :
            execute_classifyfailure(uiStepHistoryId )
        print(f"\n[ERROR] Search failed with exception: {e}")  
        print(f"[ERROR] Search duration: {elapsed_time:.2f} seconds")  

        if args.debug:  
            import traceback  
            print(f"[DEBUG] Full traceback:")  
            traceback.print_exc()  

        return 1  
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------#
if __name__ == "__main__":  
    """Entry point when script is run directly."""  
    try:  
        exit_code = main()  
        sys.exit(exit_code)  
    except Exception as e:  
        print(f"[FATAL] Unhandled exception in main: {e}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(2)
