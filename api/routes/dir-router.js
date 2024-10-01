const express = require("express");
// const bodyParser = require('body-parser')

const dirCtrl = require("../controllers/dir-ctrl");

const router = express.Router();
// router.use(bodyParser.raw({inflate:true, limit: '100kb', type: 'application/json'}));

// router.get("/history/:std_id", stdCtrl.getHistoryByID);
// router.get("/history", stdCtrl.getHistory);

// router.delete("/del_history/:std_id", stdCtrl.deleteHistory);

router.post("/create_dir", dirCtrl.createDir);
// router.put("/update_history/:std_id", stdCtrl.updateHistory);

router.get("/get_dir_detail", dirCtrl.getAllDir);
router.post("/update_dir", dirCtrl.updateDir);
router.post("/del_dir", dirCtrl.deleteDir);
// router.post("/acc", stdCtrl.createAccount);
// router.put("/acc/:std_id", stdCtrl.updateAccount);
// router.get("/acc/:std_id", stdCtrl.getAccountByID);

// router.post("/proof", stdCtrl.createProof);
// router.put("/proof/:std_id", stdCtrl.updateProof);
router.post("/get_dir_detail/input", dirCtrl.getDirByInput);

module.exports = router;
