try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    METEOR_AVAILABLE = True
    SPICE_AVAILABLE = True
except ImportError:
    print("Warning: `pycocoevalcap` is not installed. Metrics calculation will not work.")
    print("Please install it. A common way is: `pip install adeb-remedy`")
    # Define dummy classes if the import fails to avoid runtime errors
    Bleu = Meteor = Rouge = Cider = Spice = object
    METEOR_AVAILABLE = False
    SPICE_AVAILABLE = False


class CaptionMetrics:
    """
    A wrapper class to compute all standard image captioning metrics.
    It uses the official pycocoevalcap library to ensure results are
    comparable with published research.
    """
    def __init__(self, references):
        """
        Args:
            references (dict): A dictionary mapping image IDs to a list of
                               ground truth reference captions.
                               Example: {0: ["a cat", "a feline"], 1: ["a dog"]}
        """
        self.scorers = []
        
        # BLEU (always works - pure Python)
        if METEOR_AVAILABLE:
            self.scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        
        # METEOR (requires Java)
        if METEOR_AVAILABLE:
            try:
                meteor_scorer = Meteor()
                self.scorers.append((meteor_scorer, "METEOR"))
            except Exception as e:
                print(f"Warning: METEOR initialization failed (Java required): {e}")
                print("Skipping METEOR metric. Install Java to enable it.")
        
        # ROUGE (pure Python)
        if METEOR_AVAILABLE:
            self.scorers.append((Rouge(), "ROUGE_L"))
        
        # CIDEr (pure Python)
        if METEOR_AVAILABLE:
            self.scorers.append((Cider(), "CIDEr"))
        
        # SPICE (requires Java)
        if SPICE_AVAILABLE:
            try:
                spice_scorer = Spice()
                self.scorers.append((spice_scorer, "SPICE"))
            except Exception as e:
                print(f"Warning: SPICE initialization failed (Java required): {e}")
                print("Skipping SPICE metric. Install Java to enable it.")
        
        self.references = self._format_for_eval(references)

    def _format_for_eval(self, captions_dict):
        """
        Converts a dictionary of captions into the format required by the
        pycocoevalcap library.
        
        The required format is: {image_id: [{'caption': caption_string}, ...]}
        """
        formatted_captions = {}
        for img_id, captions in captions_dict.items():
            formatted_captions[img_id] = [{"caption": cap} for cap in captions]
        return formatted_captions

    def compute_scores(self, hypotheses):
        """
        Computes all captioning scores for a given set of hypotheses.

        Args:
            hypotheses (dict): A dictionary mapping image IDs to a list
                               containing a single generated caption.
                               Example: {0: ["a cat sitting"], 1: ["a dog running"]}
        
        Returns:
            dict: A dictionary containing all computed scores.
        """
        if not hasattr(Cider, 'compute_score'):
             print("Cannot compute scores because `pycocoevalcap` is not properly installed.")
             return {}
             
        formatted_hypotheses = self._format_for_eval(hypotheses)
        
        all_scores = {}
        for scorer, method in self.scorers:
            try:
                print(f'Computing {scorer.method()} score...')
                score, scores = scorer.compute_score(self.references, formatted_hypotheses)
                
                if isinstance(method, list): # For BLEU
                    for sc, scs, m in zip(score, scores, method):
                        all_scores[m] = sc
                else: # For other metrics
                    all_scores[method] = score
            except Exception as e:
                print(f"Error computing {method}: {e}")
                continue
        
        return all_scores


# Example usage:
if __name__ == '__main__':
    # --- Mock Data ---
    # Ground truth reference captions (multiple per image)
    ground_truth = {
        0: [
            "a cat is sleeping on the couch",
            "a kitty naps on the sofa",
            "there is a cat on the couch"
        ],
        1: [
            "a brown dog is running in the park",
            "a dog plays in a grassy field"
        ]
    }

    # Model-generated hypothesis captions (one per image)
    hypotheses = {
        0: ["a cat is on the couch"],
        1: ["a dog is running in a field"]
    }
    
    print("--- Testing CaptionMetrics ---")
    
    # --- Initialize the metrics calculator ---
    try:
        metrics_calculator = CaptionMetrics(ground_truth)
        
        # --- Compute the scores ---
        scores = metrics_calculator.compute_scores(hypotheses)
        
        print("\n--- Computed Scores ---")
        for metric_name, score_value in scores.items():
            print(f"{metric_name:<8}: {score_value:.4f}")
            
        assert "CIDEr" in scores
        assert "Bleu_4" in scores

    except (ImportError, NameError):
        print("\nSkipping example usage because `pycocoevalcap` is not installed.")