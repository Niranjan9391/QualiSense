def get_model2_recommendations(defect_class):
    """
    Returns 2–3 static suggestions depending on the predicted defect.
    """

    rec_map = {

        "Porosity": [
            "Increase casting pressure to reduce trapped air pockets.",
            "Reduce melt temperature slightly to improve solidification.",
            "Ensure mold is free of moisture before operation."
        ],

        "Shrinkage": [
            "Increase cooling time to allow uniform metal contraction.",
            "Reduce mold temperature to improve solidification rate.",
            "Check gating system to avoid uneven metal flow."
        ],

        "Cold Shut": 
        [
            "Increase melt temperature to improve metal fluidity.",
            "Increase flow rate to allow metal streams to merge properly.",
            "Reduce oxidation by improving furnace cleanliness."
        ],

        "Misrun":
        [
            "Increase mold temperature to improve metal flow.",
            "Increase casting pressure for complete mold filling.",
            "Ensure gate design is not restricting metal flow."
        ],

        "No Defect": 
        [
            "Process parameters are stable. Maintain current operating conditions."
        ],
    }

    # Default fallback
    return rec_map.get(defect_class, ["No specific recommendations available."])
