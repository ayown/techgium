import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import mongoose from "mongoose";

// Load env variables
dotenv.config();

// Initialize App
const app = express();
const PORT = process.env.PORT || 5000;

// --- Middlewares ---
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// --- Database Connection (MongoDB Atlas) ---
const connectDB = async () => {
	try {
		const conn = await mongoose.connect(process.env.MONGODB_URI);
		console.log(`MongoDB Connected: ${conn.connection.host}`);
	} catch (error) {
		console.error(`Error: ${error.message}`);
		process.exit(1);
	}
};

// Uncomment to connect
// connectDB();

// --- Routes ---

// 1. Health Check
app.get("/", (req, res) => {
	res.status(200).json({ message: "API is running..." });
});
// --- Error Handling Middleware ---
app.use((err, req, res, next) => {
	const statusCode = res.statusCode === 200 ? 500 : res.statusCode;
	res.status(statusCode);
	res.json({
		message: err.message,
		stack: process.env.NODE_ENV === "production" ? null : err.stack,
	});
});

// --- Start Server ---
app.listen(PORT, () => {
	console.log(
		`Server running in ${
			process.env.NODE_ENV || "development"
		} mode on port ${PORT}`
	);
});
