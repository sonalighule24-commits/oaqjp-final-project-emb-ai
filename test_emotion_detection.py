import unittest
from unittest.mock import patch, Mock
from emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    @patch("emotion_detection.requests.post")
    def test_emotion_detector(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200

        mock_response.json.return_value = {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.8,
                        "disgust": 0.1,
                        "fear": 0.2,
                        "joy": 0.05,
                        "sadness": 0.3
                    }
                }
            ]
        }

        mock_post.return_value = mock_response

        result = emotion_detector("I am really mad about this")

        self.assertEqual(result["dominant_emotion"], "anger")
        self.assertEqual(result["anger"], 0.8)
        self.assertEqual(result["disgust"], 0.1)
        self.assertEqual(result["fear"], 0.2)
        self.assertEqual(result["joy"], 0.05)
        self.assertEqual(result["sadness"], 0.3)


if __name__ == "__main__":
    unittest.main()
