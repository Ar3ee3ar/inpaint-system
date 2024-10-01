
// const fs = module.constructor._load('fs');

const fs = require('fs');

function readFiles(dirname, onError) {
  fs.readdir(dirname, function(err, filenames) {
    if (err) {
      onError(err);
      return;
    }
    filenames.forEach(function(filename) {
      console.log(dirname+filename)
    });
  });
}

// import { getAllFiles, getAllFilesSync } from 'get-all-files'

// async function read_folders(folder_path){
// //     const reg_pattern = "*.jpg";
// //     // Lazily iterate over filenames asynchronously
// //     for await (const filename of getAllFiles(folder_path)) {
// //         // Could break early on some condition and get-all-files
// //         // won't have unnecessarily accumulated the filenames in an array
// //         if(reg_pattern.exec(filename)){
// //             console.log(filename+" -> img")
// //         }
// //         else{
// //             console.log(filename)
// //         }
// //     };
// // }

// // Get array of filenames asynchronously
// console.log(await getAllFiles(`path/to/dir/or/file`).toArray())
// }
export default readFiles;