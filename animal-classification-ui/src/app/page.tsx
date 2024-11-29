"use client";

import { useState } from "react";

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!image) {
      alert("Please upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResult(data.prediction || "Error in prediction.");
    } catch (error) {
      console.log(error);
      setResult("Failed to fetch prediction.");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-black p-4">
      <h1 className="text-3xl font-bold mb-6">Animal Classifier</h1>
      <form
        onSubmit={handleSubmit}
        className="flex flex-col items-center gap-4 bg-white p-6 rounded-lg shadow-lg"
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="block w-full text-sm text-gray-500
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-full file:border-0
                     file:text-sm file:font-semibold
                     file:bg-indigo-50 file:text-indigo-700
                     hover:file:bg-indigo-100"
        />
        {imagePreview && (
          <div className="mt-4 border rounded-lg p-4">
            <img
              src={imagePreview}
              alt="Uploaded preview"
              className="max-w-lg h-auto rounded-lg"
            />
          </div>
        )}
        <button
          type="submit"
          className="px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600"
        >
          Predict
        </button>
      </form>
      {result && (
        <div className="mt-6 p-4 bg-green-100 text-green-700 rounded-lg">
          <p>Prediction: {result}</p>
        </div>
      )}
    </div>
  );
}
