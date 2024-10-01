const { setTimeout } = require("timers/promises");

const express = require("express");
const { createServer } = require("node:http");
const { join } = require("node:path");
const { Server } = require("socket.io");
const cors = require("cors");
const { glob } = require("glob");
const fs = require("node:fs");
const e = require("cors");


// const db = require("./db");
// const dirRouter = require("./routes/dir-router");
// const dirCtrl = require("./controllers/dir-ctrl");
// const { dirDetail } = require("./models/dir-models");

const app = express();
const server = createServer(app);
const io = new Server(server, {
  pingTimeout: 3000000,
  cors: {
    origin: "*", // Allow cross-origin requests from your React app
    methods: ["GET", "POST"],
  },
});


// ----------------- socket -----------------------------
var clients = {};
var client_detail = {
  node_name: "",
  sid: "",
  process: 0,
  all_img: 0,
  status: "",
  process_folder: "",
};
var selected_client = {};
// Store pending responses from clients
let pendingResponses = {};

// function getKeyByValue(object, value) {
//   return Object.keys(object).find((key) => object[key] === value);
// }
io.on("connection", async (socket) => {
  console.log("a user connected");
  if (Object.keys(clients).length == 0) {
    node_num = Object.keys(clients).length;
  } else {
    node_num =
      clients[Object.keys(clients)[Object.keys(clients).length - 1]][
        "node_name"
      ];
    node_num = node_num + 1;
  }
  client_name = node_num;
  client_detail["node_name"] = node_num;
  client_detail["sid"] = socket.id;
  clients[socket.id] = client_detail;
  client_detail = {
    node_name: "",
    sid: "",
    process: 0,
    all_img: 0,
    status: "",
    process_folder: "",
  };

  // console.log(Object.keys(clients) + " " + (Object.keys(clients).length - 1));
  // console.log(clients);

  socket.on("disconnect", () => {
    // console.log(socket.id);
    // console.log(clients[socket.id]);
    if(selected_client[socket.id]){
      selected_client[socket.id]["status"] = "stop"
    }
    const client_name = clients[socket.id]["node_name"];
    const client_sid = socket.id;
    delete clients[socket.id];
    console.log(client_name + " ( " + client_sid + " )" + "user disconnected");
    // console.log("total node in system : " + Object.keys(clients));
  });

  console.log("send message");
  // socket.emit("hello", "world");
  socket.emit("hello", clients[socket.id]);
  // await setTimeout(5000);
  // socket.emit("run", 10);
  // await setTimeout(4000);
  // // console.log("server want process")
  // socket.emit("stop_process");

  socket.on("my_response", function (message) {
    console.log(message);
  });

  socket.on("run_progress", function (message) {
    if (Number.isInteger(message)) {
      clients[socket.id]["process"] = message;
    } else {
      clients[socket.id]["status"] = message;
    }
    console.log(
      "node" +
        clients[socket.id]["node_name"] +
        " : progress: " +
        clients[socket.id]["process"] +
        "/" +
        clients[socket.id]["all_img"] +
        "\nstatus:" +
        clients[socket.id]["status"]
    );
  });

  // Handle the response from the client
  socket.on("clientResponse", (data) => {
    console.log(`Response from client ${socket.id}:`, data);
    if (pendingResponses[socket.id]) {
      // Resolve the promise when a response is received
      pendingResponses[socket.id](data);
      delete pendingResponses[socket.id]; // Cleanup after resolving
    }
  });

  // clients.push(socket)
});

// ------------------ function -----------------------------

async function callModel(pic_detail) {
  console.log(pic_detail);
  await fetch("http://localhost:5000/offline-pic", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(pic_detail),
  })
    .then((response) => {
      console.log(response.data);
      // return (response.data);
    })
    .catch((error) => {
      console.log(error);
    });
}

async function readFiles(in_dirname, file_pattern) {
  // console.log(in_dirname);
  var files = await glob(in_dirname + file_pattern, {});
  files.sort();
  // console.log(files);
  for (let i = 0; i < files.length; i++) {
    files[i] = files[i].replaceAll("\\", "/");
  }
  return files;
}

async function findGPS(in_dirname) {
  // console.log(in_dirname);
  var files = await glob(in_dirname + "*" + "_camera_position.txt", {});
  // files.sort();
  // // console.log(files);
  // for (let i = 0; i < files.length; i++) {
  //   files[i] = files[i].replaceAll("\\", "/");
  // }
  return files;
}

async function readFolders(in_dirname) {
  // console.log(in_dirname);
  var files = await glob(in_dirname + '*/*/Camera/', {});
  files.sort();
  // console.log(files);
  for (let i = 0; i < files.length; i++) {
    files[i] = files[i].replaceAll("\\", "/");
  }
  return files;
}

async function createFolder(folderName) {
  // console.log(folderName);
  try {
    if (!fs.existsSync(folderName)) {
      fs.mkdirSync(folderName);
      return "create " + folderName + " complete";
    }
    else{
      return "already create " + folderName;
    }
  } catch (err) {
    console.error(err);
  }
}

function getKeyByValue(object, value, attr) {
  if(attr == "node_name"){
    return Object.keys(object).find((key) => object[key].node_name === value);
  }
}

// Helper function to wait for a specific client's response
const waitForClientResponse = (clientId) => {
    return new Promise((resolve, reject) => {
        if (clients[clientId]) {
          pendingResponses[clientId] = resolve;

          // // Set a timeout to reject if no response is received in 10 seconds
          // setTimeout(() => {
          //     if (pendingResponses[clientId]) {
          //         delete pendingResponses[clientId]; // Cleanup
          //         reject(new Error(`Timeout: No response from client ${clientId}`));
          //     }
          // }, 10000); // 10 seconds timeout
        } else {
          reject(new Error(`Client ${clientId} not found or disconnected`));
        }
    });
}
// ----------------------- REST API -------------------------

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.sendFile(join(__dirname, "index.html"));
});

// app.get("/test_read_dir", async function (req,res){
//   var data = []
//   try {
//     const dir_file = await dirDetail.find({});
//     console.log(dir_file.length);
//     // console.log(dir_file[1]);
//     if (!dir_file) {
//       data = []
//     }
//     else{
//       data = dir_file;
//     }
//     res.send(data);
//   } catch (error) {
//     return res.status(500).json({ message: error.message });
//   }
// })

// app.post('/read-dir', async function (req, res){
//   const data = req.body;
//   var count_save = 0;
//   folder_path = await readFolders(data.input);
//   for (let i=0; i<folder_path.length; i++){
//     var data_list = await readFiles(folder_path[i]+'/', "*.jpg");
//     var have_gps = false;
//     if ((findGPS(folder_path[i] + "/")).length > 0) {
//       have_gps = true;
//     }
//     dir_body = {
//       input_path: folder_path[i] + "/",
//       output_path: folder_path[i] + "/" + "output/",
//       have_gps: have_gps,
//       process: 0,
//       all_pic: data_list.length,
//       complete: false,
//     };
//     try {
//       const dir_file = await dirDetail.findOne({
//         input_path: folder_path[i] + "/",
//       });

//       if (!dir_file) {
//         count_save = count_save + 1;
//         console.log(count_save);
//         try {
//           const dir_file = new dirDetail(dir_body);
//           dir_file.save().then(() => {
//             // return res.status(201).json({
//             //   success: true,
//             //   id: dir_file._id,
//             //   message: "directory created!",
//             // });
//           });
//           // .catch((error) => {
//           //   continue;
//           //   // console.log(error)
//           //   // return res.sendStatus(400).json({
//           //   //   error,
//           //   //   message: "directory not created!",
//           //   // });
//           // });
//         } catch (error) {
//           return res
//             .sendStatus(400)
//             .json({ success: false, error: error.message });
//         }
//       }
//       // return res.status(200).json({ success: true, data: dir_file });
//     } catch (error) {
//       return res.sendStatus(400).json({ success: false, error: error.message });
//     }
//   }
//   // console.log(count_save);
//   res.send(count_save.toString());
// })

app.post("/offline-main-gps", async function (req, res) {
  const data = req.body;
  console.log(data);
  folder_path = await readFolders(data.input);
  console.log(folder_path);
  // folder_path = [folder_path[0]];
  for (let fol=0; fol<folder_path.length; fol++){
    const responsePromises = []; // Array to hold the promises for each client response
    console.log(folder_path[fol]);
    var data_list = await readFiles(folder_path[fol] + "/", "*.jpg");
    var line_name = (folder_path[fol].split('/')).at(3)
    var save_fol = folder_path[fol] + "/" + line_name+"_"+data.save_fol + "/"; // add date to output file name ex. output_01102024
    var gps_path = await findGPS(folder_path[fol] + "/");
    // console.log(folder_path[i] + "/");
    // console.log(gps_path);
    if (gps_path.length > 0) {
      if (!fs.existsSync(save_fol)) {
        var res_createFolder = await createFolder(save_fol);
        console.log(res_createFolder);
      } else {
        for (let pic_path = 0; pic_path < data_list.length; pic_path++) {
          // console.log(data_list[pic_path]);
          pic_full_name = data_list[pic_path].split("/").at(-1);
          // console.log(pic_full_name);
          pic_name = pic_full_name.slice(0,-4);
          // console.log(pic_name);
          // break;
          if(fs.existsSync(save_fol+pic_name+'_O.jpg')){
            data_list.splice(pic_path, 1); // Remove element at index i
            pic_path--; // Adjust index to account for removed element
          }
        }
      }
      // data_list = [];
      if(data_list.length > 0){
        io.emit("pre_run", folder_path[fol] + "/");
        // console.log(data_list.at(0));
        // console.log(data_list.at(1));
        // console.log(data_list.length);
        var client_send_command = {};
        var pic_detail = {
          main_path: folder_path[fol] + "/",
          input: [],
          save_fol: save_fol,
        };

        // var data_list = await readFiles(data.input, "*.jpg");
        // for (let i = 0; i < data_list.length; i++) {
        //   pic_detail["input"].push(data_list[i]);
        //   // await callModel(pic_detail);
        //   // break;
        // }
        // data_list = []
        // for (let j = 0; j<100; j++){
        //   data_list.push(j)
        // }
        divide_work = Object.keys(selected_client).length;
        console.log("divide work");
        if (divide_work > 0) {
          work = ~~(data_list.length / divide_work);
          // console.log(Object.keys(clients))
          for (let i = 0; i < Object.keys(selected_client).length - 1; i++) {
            pic_detail["input"] = data_list.slice(work * i, work * (i + 1));
            pic_detail["status"] = "assign task";
            selected_client[Object.keys(selected_client)[i]]["all_img"] =
              work * (i + 1) - work * i;
            selected_client[Object.keys(selected_client)[i]]["process"] = 0;
            selected_client[Object.keys(selected_client)[i]]["process_folder"] =
              folder_path[fol];
            client_send_command[Object.keys(selected_client)[i]] = pic_detail;
            // io.to(Object.keys(selected_client)[i]).emit("run", pic_detail);
          }
          // console.log(work + " * " + divide_work);
          pic_detail["input"] = data_list.slice(work * (divide_work - 1));
          pic_detail["status"] = "assign task";
          selected_client[
            Object.keys(selected_client)[
              Object.keys(selected_client).length - 1
            ]
          ]["all_img"] = data_list.length - work * (divide_work - 1);
          selected_client[
            Object.keys(selected_client)[
              Object.keys(selected_client).length - 1
            ]
          ]["process"] = 0;
          selected_client[
            Object.keys(selected_client)[
              Object.keys(selected_client).length - 1
            ]
          ]["process_folder"] = folder_path[fol];
          client_send_command[
            Object.keys(selected_client)[
              Object.keys(selected_client).length - 1
            ]
          ] = pic_detail;
          // io.to(Object.keys(selected_client)[Object.keys(selected_client).length - 1]).emit("run", pic_detail);

          for (let j = 0; j < Object.keys(selected_client).length; j++) {
            io.to(Object.keys(selected_client)[j]).emit(
              "run",
              client_send_command[Object.keys(selected_client)[j]]
            );
            // Step 2: Prepare to wait for the response from this client
            const responsePromise = waitForClientResponse(
              Object.keys(selected_client)[j]
            );
            responsePromises.push(responsePromise); // Collect the promise
          }
          // res.send(selected_client);

          // Step 3: Wait for all clients to respond
          const allResponses = await Promise.all(responsePromises);
          console.log("Received all responses:", allResponses);
          // res.send(selected_client);

          console.log(
            "Received responses from " + (fol + 1) + " command:",
            folder_path[fol]
          );
        } else {
          // for (let i = 0; i < Object.keys(clients).length - 1; i++){
          //   pic_detail["status"] = "assign task";
          // }
          res.send("not have node in system");
        }
      }
      else{
        console.log("already process", folder_path[fol]);
        for (let k = 0; k < Object.keys(selected_client).length; k++) {
          selected_client[Object.keys(selected_client)[k]]["status"] = "wait";
        }
      }
    }
    // await setTimeout(50000);
    await io.on("finish", () => {
      console.log("already finish");
    });
  }
  for (let k = 0; k < Object.keys(selected_client).length; k++) {
    selected_client[Object.keys(selected_client)[k]]["status"] = "complete";
  }
});

app.post("/offline-folder-gps", async function (req, res) {
  const data = req.body;
  io.emit("pre_run", data.input);
  console.log(data)
  var res_createFolder = await createFolder(data.save_fol);
  console.log(res_createFolder);

  var pic_detail = {
    main_path: data.input,
    input: [],
    save_fol: data.save_fol,
  };

  var data_list = await readFiles(data.input, "*.jpg");
  // for (let i = 0; i < data_list.length; i++) {
  //   pic_detail["input"].push(data_list[i]);
  //   // await callModel(pic_detail);
  //   // break;
  // }
  // data_list = []
  // for (let j = 0; j<100; j++){
  //   data_list.push(j)
  // }
  divide_work = Object.keys(selected_client).length;
  if (divide_work > 0){
    work = ~~(data_list.length / divide_work);
    // console.log(Object.keys(clients))
    for (let i = 0; i < Object.keys(selected_client).length - 1; i++) {
      pic_detail["input"] = data_list.slice(work * i, work * (i + 1));
      pic_detail["status"] = "assign task";
      selected_client[Object.keys(selected_client)[i]]["all_img"] = work * (i + 1) - work * i;
      selected_client[Object.keys(selected_client)[i]]["process"] = 0;
      io.to(Object.keys(selected_client)[i]).emit("run", pic_detail);
    }
    // console.log(work + " * " + divide_work);
    pic_detail["input"] = data_list.slice(work * (divide_work - 1));
    pic_detail["status"] = "assign task";
    selected_client[Object.keys(selected_client)[Object.keys(selected_client).length - 1]]["all_img"] =
      data_list.length - work * (divide_work - 1);
    selected_client[Object.keys(selected_client)[Object.keys(selected_client).length - 1]]["process"] = 0;
    io.to(
      Object.keys(selected_client)[Object.keys(selected_client).length - 1]
    ).emit("run", pic_detail);
    res.send(selected_client);
  }
  else{
    // for (let i = 0; i < Object.keys(clients).length - 1; i++){
    //   pic_detail["status"] = "assign task";
    // }
    res.send("not have node in system");
  }
});

app.get("/node_detail", async function (req, res) {
  res.send(clients);
});

app.get("/task", async function (req, res) {
  res.send(selected_client);
});

app.get("/stop_all", async function (req, res) {
  io.emit("stop_process");
  res.send("stop_all")
});

app.post("/want_node", async function (req, res) {
  const data = req.body;
  selected_client = {};
  console.log(data)
  for (let i = 0; i < Object.keys(clients).length; i++) {
    // console.log(Object.keys(clients).find((key) => clients[key].node_name === data[i]));
    // key = getKeyByValue(clients, data[i], "node_name");
    // clients[key].status = "process";
    node_name = clients[Object.keys(clients)[i]]["node_name"]
    if(data.includes(node_name)){
      clients[Object.keys(clients)[i]].status = "process";
      selected_client[Object.keys(clients)[i]] = clients[Object.keys(clients)[i]]
    }
    else{
      clients[Object.keys(clients)[i]].status = "";
    }
  }
  console.log(selected_client);
  res.send(selected_client);
});



// ------------------------ REST API (db) ---------------------
// db.on("error", console.error.bind(console, "MongoDB connection error:"));
// app.use("/api", dirRouter);

const port = 5000
server.listen(port, () => {
  console.log("server running at http://localhost:" + port);
});

// const net = require("net");

// const server = net.createServer();

// clients = []

// // server.on("connection", (socket) => {
// //   console.log("A new connection to the server");

// //   socket.on("data", (data) => {
// //     clients.map((s) => {
// //         s.write(data)
// //     })
// //   });

// //   clients.push(socket)
// // });

// server.listen(3008, "127.0.0.1", () => {
//   console.log("opened server on", server.address());
// });

// const express = require("express");
// const app = express();
// const http = require("http");
// const server = http.createServer(app);

// app.get("/", (req, res) => {
//   res.send("<h1>Hello world</h1>");
// });

// server.listen(3000, "127.0.0.1", () => {
//   console.log("listening on *:3000");
// });
