This project utilizes several machine learning models to analyze artwork:

1. A fine-tuned MobileNet model verifies the artwork's origin, determining if it was created by a human or generated by AI.
2. Another fine-tuned MobileNet model predicts the artistic style of the piece and the emotions it evokes.
3. The original BLIP model generates a general description of the painting.
4. A fine-tuned BLIP model generates a description focusing specifically on the emotional content of the artwork.

The outputs from these individual models are then passed to the GigaChat API to synthesize the information and generate a final, well-structured, and engaging description of the artwork.
