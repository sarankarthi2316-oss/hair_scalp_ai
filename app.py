import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from services.rag_service import generate_routine
from services.tabular_service import predict_hair_loss_risk
from services.vision_service import (
    predict_baldness_density,
    predict_scalp_disease,
)

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
  return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
  try:
    # 1. Validate Image Uploads
    if "scalp_image" not in request.files or "hair_image" not in request.files:
      return (
          jsonify(
              {"error": "Please provide both scalp and hair/head images."}
          ),
          400,
      )

    scalp_img = request.files["scalp_image"]
    hair_img = request.files["hair_image"]
    form_data = request.form.to_dict()

    if scalp_img.filename == "" or hair_img.filename == "":
      return (
          jsonify({"error": "Selected files must be valid image files."}),
          400,
      )

    # 2. Run Computer Vision Models
    disease_result = predict_scalp_disease(scalp_img)
    density_result = predict_baldness_density(hair_img)

    # 3. Run Tabular Questionnaire Model
    risk_score = predict_hair_loss_risk(form_data)

    # 4. Generate Personalized Routine via RAG
    routine = generate_routine(
        disease=disease_result["condition"],
        density=density_result["density_level"],
        risk_score=risk_score,
        nutrition=form_data.get("nutritional_deficiencies", "None"),
        stress=form_data.get("stress", "Moderate"),
        medical_condition=form_data.get("medical_condition", "None"),
    )

    # 5. Return JSON payload
    return jsonify({
        "status": "success",
        "scalp_condition": disease_result,
        "hair_density": density_result,
        "hair_loss_risk": risk_score,
        "routine": routine,
    })

  except Exception as e:
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  app.run(debug=True, host="127.0.0.1", port=5000)