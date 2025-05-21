import axios from 'axios'

const api = axios.create({
    baseURL: 'http://localhost:5000/',
})

// export const createProcess = body => api.post(`/offline-folder`,body);
export const createProcessGPS = (body) => api.post(`/offline-main-gps`, body);
export const createProcessGAN = (body) => api.post(`/offline-main-gan`, body);
export const createProcessGANnakorn = (body) => api.post(`/offline-main-gan-nakorn`, body);
export const stopProcess = () => api.get(`/stop_all`);
export const getNodeDetail = () => api.get(`/node_detail`);
export const setNode = (body) => api.post(`/want_node`, body);
// export const getHistoryByID = std_id => api.get(`/history/${std_id}`);
// export const updateHistory = (std_id,body) => api.put(`/update_history/${std_id}`,body);

// export const getAccountByID = std_id => api.get(`/acc/${std_id}`);

// export const getProofByID = std_id => api.get(`/proof/${std_id}`);
// export const updateProof = (std_id,body) => api.put(`/proof/${std_id}`,body);
// export const createProof = body => api.post(`/proof`,body);

const apis = {
  // createProcess,
  createProcessGAN,
  createProcessGPS,
  createProcessGANnakorn,
  stopProcess,
  getNodeDetail,
  setNode,
};

export default apis