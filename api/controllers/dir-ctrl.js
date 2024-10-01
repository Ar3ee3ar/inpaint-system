const { dirDetail } = require("../models/dir-models");

// controller for => History and subj


const createDir = (req, res) => {
  const body = req.body;

  if (!body) {
    return res.status(400).json({
      success: false,
      error: "You must provide a directory",
    });
  }

  try {
    const dir_file = new dirDetail(body);
    dir_file
      .save()
      .then(() => {
        return res.status(201).json({
          success: true,
          id: dir_file._id,
          message: "directory created!",
        });
      })
      .catch((error) => {
        return res.status(400).json({
          error,
          message: "directory not created!",
        });
      });
  } catch (error) {
    return res.status(400).json({ success: false, error: error.message });
  }
};

const updateDir = async (req, res) => {
  // console.log(typeof(req))
  const body = req.body;
  // console.log(body.json)

  if (!body) {
    return res.status(400).json({
      success: false,
      error: "You must provide a body to update",
    });
  }
  try {
    // console.log(body)
    const dir_file = await dirDetail.findOneAndUpdate(
      { input_path: body.input_path },
      body,
      {
        new: true,
      }
    );
    // console.log(history_file)
    if (!dir_file) {
      return res.status(404).json({ message: `directory not found` });
    }
    return res.status(200).json({ success: true, data: dir_file });
  } catch (error) {
    return res.status(400).json({ success: false, error: error.message });
  }
};

const deleteDir = async (req, res) => {
  const body = req.body;
  try {
    const dir_file = await dirDetail.findOneAndDelete({
      input_path: body.input_path,
    });
    if (!dir_file) {
      return res.status(404).json({ message: `directory not found` });
    }
    return res.status(200).json({ success: true, data: dir_file });
  } catch (error) {
    return res.status(400).json({ success: false, error: error.message });
  }
};

// const getHistoryByID = async (req, res) => {
//   try {
//     const history_file = await history.aggregate([
//       {
//         $lookup: {
//           from: "all_subj_info",
//           localField: "course.subj_ID",
//           foreignField: "_id",
//           as: "subj_info",
//         },
//       },
//       {
//         $match: { _id: req.params.std_id },
//       },
//     ]);

//     if (!history_file) {
//       return res.status(404).json({ message: `Student not found` });
//     }
//     return res.status(200).json({ success: true, data: history_file });
//   } catch (error) {
//     return res.status(400).json({ success: false, error: error.message });
//   }
// };

// const getHistory = async (req, res) => {
//   try {
//     const history_file = await history.aggregate([
//       {
//         $lookup: {
//           from: "all_subj_info",
//           localField: "course.subj_ID",
//           foreignField: "_id",
//           as: "subj_info",
//         },
//       },
//     ]);
//     // console.log(data);
//     if (!history_file) {
//       return res.status(404).json({ message: `Student not found` });
//     }
//     return res.status(200).json({ success: true, data: history_file });
//   } catch (error) {
//     return res.status(500).json({ message: error.message });
//   }
// };

const getAllDir = async (req, res) => {
  try {
    const dir_file = await dirDetail.find({});
    // console.log(dir_file.length);
    // console.log(dir_file[1]);
    if (!dir_file) {
      return res.status(404).json({ message: `directory not found` });
    }
    return res.status(200).json({ success: true, data: dir_file });
  } catch (error) {
    return res.status(500).json({ message: error.message });
  }
};

// // controller for => Account
// const createAccount = (req, res) => {
//   const body = req.body;
//   // console.log(body)

//   if (!body) {
//     return res.status(400).json({
//       success: false,
//       error: "You must provide a movie",
//     });
//   }

//   try {
//     const account_file = new account(body);
//     account_file
//       .save()
//       .then(() => {
//         return res.status(201).json({
//           success: true,
//           id: account_file._id,
//           message: "Account created!",
//         });
//       })
//       .catch((error) => {
//         return res.status(400).json({
//           error,
//           message: "Account not created!",
//         });
//       });
//   } catch (error) {
//     return res.status(400).json({ success: false, error: error.message });
//   }
// };

// const updateAccount = async (req, res) => {
//   // console.log(typeof(req))
//   const body = req.body;
//   // console.log(body.json)

//   if (!body) {
//     return res.status(400).json({
//       success: false,
//       error: "You must provide a body to update",
//     });
//   }
//   try {
//     // console.log(body)
//     const account_file = await account.findOneAndUpdate(
//       { _id: req.params.std_id },
//       body,
//       {
//         new: true,
//       }
//     );
//     // console.log(history_file)
//     if (!account_file) {
//       return res.status(404).json({ message: `Account not found` });
//     }
//     return res.status(200).json({ success: true, data: account_file });
//   } catch (error) {
//     return res.status(400).json({ success: false, error: error.message });
//   }
// };

// const getAccountByID = async (req, res) => {
//   try {
//     const account_file = await account.findOne({ _id: req.params.std_id });

//     if (!account_file) {
//       return res.status(404).json({ message: `Account not found` });
//     }
//     return res.status(200).json({ success: true, data: account_file });
//   } catch (error) {
//     return res.status(400).json({ success: false, error: error.message });
//   }
// };
// // controller for => ProofTran
// const createProof = (req, res) => {
//   const body = req.body;
//   console.log(body);

//   if (!body) {
//     return res.status(400).json({
//       success: false,
//       error: "You must provide a movie",
//     });
//   }

//   try {
//     const trans_file = new trans(body);
//     trans_file
//       .save()
//       .then(() => {
//         return res.status(201).json({
//           success: true,
//           id: trans_file._id,
//           message: "Proof created!",
//         });
//       })
//       .catch((error) => {
//         return res.status(400).json({
//           error,
//           message: "Proof not created!",
//         });
//       });
//   } catch (error) {
//     return res.status(400).json({ success: false, error: error.message });
//   }
// };

// const updateProof = async (req, res) => {
//   // console.log(typeof(req))
//   const body = req.body;
//   // console.log(body.json)

//   if (!body) {
//     return res.status(400).json({
//       success: false,
//       error: "You must provide a body to update",
//     });
//   }
//   try {
//     // console.log(body)
//     const trans_file = await trans.findOneAndUpdate(
//       { _id: req.params.std_id },
//       body,
//       {
//         new: true,
//       }
//     );
//     // console.log(history_file)
//     if (!trans_file) {
//       return res.status(404).json({ message: `Proof not found` });
//     }
//     return res.status(200).json({ success: true, data: trans_file });
//   } catch (error) {
//     return res.status(400).json({ success: false, error: error.message });
//   }
// };

const getDirByInput = async (req, res) => {
  const body = req.body;
  try {
    const dir_file = await dirDetail.findOne({
      input_path: body.input_path,
    });

    if (!dir_file) {
      return res.status(404).json({ message: `directory not found` });
    }
    return res.status(200).json({ success: true, data: dir_file });
  } catch (error) {
    return res.status(400).json({ success: false, error: error.message });
  }
};

module.exports = {
  createDir,
  // updateHistory,
  // deleteHistory,
  // getHistory,
  // getHistoryByID,
  getAllDir,
  // createAccount,
  // updateAccount,
  // getAccountByID,
  // createProof,
  // updateProof,
  getDirByInput,
  updateDir,
  deleteDir,
};
