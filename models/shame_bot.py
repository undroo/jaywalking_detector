import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from google import genai
import logging
from pathlib import Path
import sys
import pyttsx3
import cv2
import base64
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ShameBot:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.logger = logging.getLogger(__name__)
        self.engine = pyttsx3.init()
        self._setup_gemini()

    def _setup_gemini(self) -> None:
        """Set up the Gemini API with the provided key."""
        try:
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                
                self.logger.info("Gemini API configured successfully")
            else:
                self.logger.warning("API key not set, Gemini API will not be available")
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise

    def _prepare_image_for_gemini(self, image):
        """
        Convert OpenCV image to format suitable for Gemini API.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            Image data in the format expected by Gemini API
        """
        try:
            # Convert BGR to RGB (OpenCV uses BGR, but most APIs expect RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', rgb_image)
            
            # Convert to base64
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "mime_type": "image/jpeg",
                "data": image_data
            }
        except Exception as e:
            self.logger.error(f"Error preparing image for Gemini: {str(e)}")
            raise

    def create_roast(self, information: str, image):
        """
        Create a roast using Gemini API with image and text prompt.
        
        Args:
            information: Information about the jaywalking incident
            image: OpenCV image (numpy array)
            
        Returns:
            Generated roast text
        """
        try:
            # Prepare the image for Gemini API
            # prepared_image = self._prepare_image_for_gemini(image)

            
            prompt = f'''
                You are a chatbot built to roast people who are jaywalking across the street.
                Keep the roasts short, approximately one sentence, it will be used as text to speech.
                Use the image to help you create the roast.
                Be mean but funny

                Here is information on the jaywalk:
                {information}
                
                Generate a short, funny roast about this jaywalker.
            '''
            
            # Generate response using the model
            response = self.client.models.generate_content(
                contents=[image, prompt],
                model="gemini-2.5-flash",   
            
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Hey there, jaywalker! Maybe try using the crosswalk next time?"
                
        except Exception as e:
            self.logger.error(f"Error generating roast: {str(e)}")
            return "Oops! Looks like someone forgot how to use a crosswalk!"

    def speak(self, text: str):
        """Convert text to speech."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {str(e)}")
            print(f"Text to speech failed: {text}")

    def roast(self, information: str, image):
        """
        Generate and speak a roast for a jaywalker.
        
        Args:
            information: Information about the jaywalking incident
            image: OpenCV image (numpy array)
        """
        try:
            roast = self.create_roast(information, image)
            print(f"Roast: {roast}")
            self.speak(roast)
        except Exception as e:
            self.logger.error(f"Error in roast method: {str(e)}")
            print("Failed to generate roast")


def main():
    """Test the ShameBot functionality."""
    try:
        shameBot = ShameBot()
        
        # Test with an image
        image_path = "test_images/image.png"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            if image is not None:
                print(f"Image loaded successfully. Size: {image.size}")
                shameBot.roast("", image)
            else:
                print("Failed to load image")
        else:
            print(f"Image file not found: {image_path}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
