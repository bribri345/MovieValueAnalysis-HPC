# -*- coding: utf-8 -*-
"""
Movie Value Analysis with Schwartz's Theory
This version uses a Hugging Face Transformers model (e.g., Llama-3.1-Centaur-70B),
processes movies in a single batch, and is ready for deployment on Hugging Face Spaces.
It emphasizes secure handling of the Hugging Face token via environment variables.
"""
import pandas as pd
import json
import time
import gradio as gr
import os
import re
import logging # Import logging

# Import for Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
SCHWARTZ_19_VALUES = {
    "Self-Direction—Thought": "Freedom to cultivate one’s own ideas and abilities.",
    "Self-Direction—Action": "Freedom to determine one’s own actions.",
    "Stimulation": "Excitement, novelty, and challenge in life.",
    "Hedonism": "Pleasure and sensuous gratification.",
    "Achievement": "Personal success through demonstrating competence according to social standards.",
    "Power—Dominance": "Power through exercising control over people.",
    "Power—Resources": "Power through control of material and social resources.",
    "Face": "Security and power achieved by maintaining one’s public image and avoiding humiliation.",
    "Security—Personal": "Safety in one’s immediate environment.",
    "Security—Societal": "Safety and stability in the wider society.",
    "Tradition": "Maintaining and preserving cultural, family, or religious traditions.",
    "Conformity—Rules": "Compliance with rules, laws, and formal obligations.",
    "Conformity—Interpersonal": "The avoidance of upsetting or harming other people.",
    "Humility": "Recognizing one’s insignificance in the larger scheme of things.",
    "Benevolence—Dependability": "Being a reliable and trustworthy member of one's in-group.",
    "Benevolence—Caring": "Devotion to the welfare of in-group members.",
    "Universalism—Concern": "A commitment to equality, justice, and protection for all people.",
    "Universalism—Nature": "The preservation of the natural environment.",
    "Universalism—Tolerance": "The acceptance and understanding of those who are different from oneself."
}

# Define the order of values for the CSV output
VALUE_HEADERS = list(SCHWARTZ_19_VALUES.keys())

# --- Model Loading and Configuration ---
def load_model():
    """
    Loads the Mistral model from a local path.
    """
    # Define the local path to the downloaded model
    model_id = "/home/tcwong383/movie_value_analysis/Mistral-7B-Instruct-v0.2"

    try:
        # Determine device for loading (0 for first GPU, -1 for CPU)
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Attempting to load model '{model_id}' on device: {'GPU' if device != -1 else 'CPU'}")

        # Load tokenizer and model from the local path. Removed 'token=hf_token'.
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        model.eval() # Set model to evaluation mode

        # Create a Hugging Face pipeline for text generation
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            return_full_text=False
        )
        logger.info(f"Model '{model_id}' configured successfully on device: {model.device}")
        return text_generator
    except Exception as e:
        logger.error(f"Failed to load model '{model_id}'. Error: {e}", exc_info=True)
        raise gr.Error(f"Failed to load model '{model_id}'. Please ensure the model ID is correct, "
                      f"you have sufficient GPU memory (70B models need a lot!), "
                      f"and required libraries are installed (e.g., bitsandbytes, accelerate). "
                      f"Error details: {e}")

generator = None
try:
    generator = load_model()
except Exception as e:
    logger.critical(f"Application start-up failed: Model could not be loaded. Analysis functionality will be unavailable. Error: {e}")

# --- Core Logic Functions ---
def generate_model_batch_prompt(movie_titles: list) -> str:
    """
    Creates a prompt for the Llama model to process a batch of movies,
    emphasizing strict JSON output with all 19 values.
    """
    movie_list_str = "\n".join([f'- "{title}"' for title in movie_titles])
    value_list_csv = ",".join(VALUE_HEADERS)

    return f"""
You are an expert at analyzing movie themes and character journeys to assess their influence on human values.

Task: For each movie in the provided list, analyze its themes, characters, plot, and message.
Assign a rating from -5 to +5 for each of the 19 Schwartz's Basic Human Values based on the movie’s influence.
-5 indicates a strong decrease in the value.
0 indicates no change.
+5 indicates a strong increase.

Consider the Protagonist’s Journey (does it reflect or challenge a value? e.g., gaining independence might increase Self-Direction—Action), Plot Resolution (what lessons are conveyed? e.g., justice prevailing might increase Universalism—Concern), Themes (what ideas are promoted? e.g., ambition might increase Achievement), and Character Portrayals (do actions reflect or contradict values? e.g., selfishness might decrease Benevolence—Caring).

The 19 values and their brief descriptions are:
{json.dumps(SCHWARTZ_19_VALUES, indent=2)}

Movies to analyze:
{movie_list_str}

Your response MUST be ONLY a single valid JSON object.
The JSON object must have a single key "movie_ratings".
The value of "movie_ratings" MUST be a list of JSON objects, one for each movie.
Each movie object MUST have a "Movie Title" key and 19 additional keys, one for each value, in the exact order: {value_list_csv}.
All ratings MUST be integers between -5 and +5.

Example of the REQUIRED JSON structure:
{{
  "movie_ratings": [
    {{
      "Movie Title": "The Lion King",
      "Self-Direction—Thought": 2,
      "Self-Direction—Action": 3,
      "Stimulation": 3,
      "Hedonism": -2,
      "Achievement": 2,
      "Power—Dominance": -3,
      "Power—Resources": -1,
      "Face": 3,
      "Security—Personal": 1,
      "Security—Societal": 2,
      "Tradition": 5,
      "Conformity—Rules": 2,
      "Conformity—Interpersonal": 4,
      "Humility": 3,
      "Benevolence—Dependability": 4,
      "Benevolence—Caring": 3,
      "Universalism—Concern": 2,
      "Universalism—Nature": 4,
      "Universalism—Tolerance": 3
    }},
    {{
      "Movie Title": "Another Movie Title",
      "Self-Direction—Thought": 0,
      "Self-Direction—Action": 0,
      "Stimulation": 0,
      "Hedonism": 0,
      "Achievement": 0,
      "Power—Dominance": 0,
      "Power—Resources": 0,
      "Face": 0,
      "Self-Direction—Thought": 0,
      "Self-Direction—Action": 0,
      "Stimulation": 0,
      "Hedonism": 0,
      "Achievement": 0,
      "Power—Dominance": 0,
      "Power—Resources": 0,
      "Face": 0,
      "Security—Personal": 0,
      "Security—Societal": 0,
      "Tradition": 0,
      "Conformity—Rules": 0,
      "Conformity—Interpersonal": 0,
      "Humility": 0,
      "Benevolence—Dependability": 0,
      "Benevolence—Caring": 0,
      "Universalism—Concern": 0,
      "Universalism—Nature": 0,
      "Universalism—Tolerance": 0
    }}
  ]
}}
"""

def get_batch_movie_ratings(movie_titles: list) -> list:
    """
    Gets value ratings for a list of movies in a single API call using the Transformers pipeline,
    with robust parsing and retries.
    """
    if generator is None:
        logger.error("Attempted to call get_batch_movie_ratings but the model (generator) is not loaded.")
        raise gr.Error("Model is not loaded. Please ensure model loading was successful at startup.")

    prompt = generate_model_batch_prompt(movie_titles)
    logger.debug(f"Sending prompt to model:\n{prompt}")

    for attempt in range(5):
        try:
            outputs = generator(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            generated_text = outputs[0]['generated_text'] if outputs else ""

            logger.info(f"Raw model response (attempt {attempt+1}, first 500 chars):\n{generated_text[:500]}...")

            match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if not match:
                logger.warning(f"Attempt {attempt + 1}: No JSON object found in model output. Retrying...")
                time.sleep(10)
                continue

            json_text = match.group(0)
            data = json.loads(json_text)

            ratings = data.get("movie_ratings", [])
            if not isinstance(ratings, list):
                raise ValueError("The 'movie_ratings' key in JSON response is not a list. Model output format issue.")

            processed_ratings = []
            for movie_data in ratings:
                if not isinstance(movie_data, dict):
                    logger.warning(f"Skipping non-dict movie entry in model response: {movie_data}")
                    continue

                movie_title = movie_data.get("Movie Title")
                if not movie_title:
                    logger.warning(f"Skipping movie entry with no 'Movie Title' key: {movie_data}")
                    continue

                clean_movie_data = {"Movie Title": movie_title}
                for value_name in VALUE_HEADERS:
                    rating = movie_data.get(value_name)
                    if isinstance(rating, (int, float)):
                        rating = int(rating)
                        if -5 <= rating <= 5:
                            clean_movie_data[value_name] = rating
                        else:
                            clean_movie_data[value_name] = None
                            logger.warning(f"Warning: Rating for '{movie_title}' - '{value_name}' ({rating}) out of -5 to +5 range. Setting to None.")
                    else:
                        clean_movie_data[value_name] = None
                        logger.warning(f"Warning: Rating for '{movie_title}' - '{value_name}' missing or not an integer. Setting to None.")
                processed_ratings.append(clean_movie_data)

            if not processed_ratings:
                raise ValueError("No valid movie ratings found after processing JSON response.")
            return processed_ratings

        except json.JSONDecodeError as e:
            logger.error(f"Attempt {attempt + 1} failed (JSON parsing error): {e} - Raw text start:\n{generated_text[:500]}...", exc_info=True)
            time.sleep(10)
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed due to an unexpected error: {e}", exc_info=True)
            time.sleep(10)
    logger.error("All attempts to get movie ratings failed. Returning empty results.")
    return None

def process_movie_list(csv_file, progress=gr.Progress()):
    """
    Processes a list of movies from a CSV in a single batch and returns a DataFrame.
    """
    if csv_file is None:
        raise gr.Error("Please upload a movie list CSV file.")
    try:
        df = pd.read_csv(csv_file.name)
        if 'Movie Title' not in df.columns:
            raise gr.Error("CSV must contain a 'Movie Title' column.")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}", exc_info=True)
        raise gr.Error(f"Failed to read CSV file: {e}")

    movie_titles = [str(title).strip() for title in df['Movie Title'] if pd.notna(title) and str(title).strip()]
    if not movie_titles:
        raise gr.Error("No valid movie titles found in the CSV. Please check the 'Movie Title' column.")

    progress(0.5, desc=f"Analyzing {len(movie_titles)} movies in one batch. This may take a moment, depending on model and movie count...")

    results = get_batch_movie_ratings(movie_titles)

    if results:
        output_df = pd.DataFrame(results)
        final_columns = ['Movie Title'] + VALUE_HEADERS

        output_df = output_df.reindex(columns=final_columns)

        progress(1.0, desc="Analysis complete!")
        return output_df
    else:
        gr.Warning("The model failed to return ratings for any movies. This could be due to model errors, invalid output, or resource limitations. Please check the movie titles or try again, or consult the logs for more details.")
        return pd.DataFrame(columns=['Movie Title'] + VALUE_HEADERS)

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Movie Value Analysis") as demo:
    gr.Markdown("# 🎬 Movie Value Analysis (Batch Mode)")
    gr.Markdown("""
    Upload a CSV with a 'Movie Title' column to analyze movies based on Schwartz's 19 basic human values
    using the Llama-3.1-Centaur-70B model. **Note: This model requires significant GPU resources and
    your Hugging Face API token (`HF_TOKEN`) set as an environment variable for access to gated models.**
    """)
    with gr.Row():
        movie_csv_input = gr.File(label="Upload Movie List CSV", file_types=[".csv"])
    
    if generator is None:
        gr.Markdown(
            """
            <p style="color: red; font-weight: bold;">
            Warning: Model failed to load at startup. Analysis functionality will not work.
            Please check your environment variables (HF_TOKEN) and GPU resources.
            Consult the application logs for detailed error messages.
            </p>
            """,
            label="Model Load Warning"
        )
    
    analyze_button = gr.Button("Analyze Movies in One Batch", variant="primary", interactive=(generator is not None))
    gr.Markdown("## 📊 Analysis Results")
    output_dataframe = gr.DataFrame(label="Movie Ratings", wrap=True)

    analyze_button.click(
        fn=process_movie_list,
        inputs=[movie_csv_input],
        outputs=[output_dataframe],
        api_name="analyze_movies"
    )

if __name__ == "__main__":
    demo.launch()
