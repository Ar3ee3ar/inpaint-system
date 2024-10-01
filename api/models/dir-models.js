const mongoose = require("mongoose");
const Schema = mongoose.Schema;

const dirSchema = new Schema(
  {
    // _id: { type: Number, required: true },
    // std_id: {type:String, required:true},
    input_path: { type: String, required: true, unique: true },
    output_path: { type: String, required: true },
    have_gps: { type: Boolean, required: true },
    process: { type: Number, required: true },
    all_pic: { type: Number, required: true },
    complete: { type: Boolean, required: true },
  },
  { collection: "dir" }
);

// Creating model objects
const dirDetail = mongoose.model("dirDetail", dirSchema);

// Exporting our model objects
module.exports = {
  dirDetail,
};
