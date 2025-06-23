// api.js 
import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  withCredentials: false,
});

export async function uploadImages({ original, mask, sketch }) {
  const formData = new FormData();
  formData.append("original", original);
  formData.append("mask", mask);
  formData.append("sketch", sketch);
  const response = await api.post("/api/v1/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export async function generateInpaint() {
  const response = await api.post("/api/v1/inpaint");
  return response.data;
} 