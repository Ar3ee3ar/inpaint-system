import React, { Component, useState, useEffect } from "react";
import {
  Button,
  Checkbox,
  Form,
  Input,
  Divider,
  Flex,
  Segmented,
  Progress,
  Col,
  Row,
  Modal,
  Card,
  Space,
} from "antd";
import { CarOutlined, HomeOutlined } from "@ant-design/icons";

import api, { getNodeDetail, stopProcess } from "../axios_api";
// import readFiles from '../tools/read_file';

// import type { FormProps } from 'antd';

let nextId = 0;
const CheckboxGroup = Checkbox.Group;
var all_result = {};
var result_detail = {
  // id: "",
  process: 0,
  all_img: 0,
  status: "",
};

export const GPSHeader = () => {
  return (
    <h1 style={{ color: "white", textAlign: "left", margin: 0 }}>
      <Space>
        <CarOutlined />
        Remove car : GPS
      </Space>
    </h1>
  );
};

export const GPSInpaint = () => {
  // const [dataTest, setDataTest] = useState("");
  const [inputPath, setInputPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [resAllResult, setResAllResult] = useState({});
  const [resFolProgress, setResFolProgress] = useState({});
  const [resNodeName, setResNodeName] = useState([]);
  const [resProgress, setResProgress] = useState([]);
  const [resAllPic, setAllPic] = useState([]);
  const [resStatus, setResStatus] = useState([]);
  const [useNode, setUseNode] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [checkedList, setCheckedList] = useState([]);
  const checkAll = useNode.length === checkedList.length;
  const indeterminate =
    checkedList.length > 0 && checkedList.length < useNode.length;
  const onChange = (list) => {
    setCheckedList(list);
  };
  const onCheckAllChange = (e) => {
    setCheckedList(e.target.checked ? useNode : []);
  };

  const showModal = () => {
    setIsModalOpen(true);
  };
  const handleOk = () => {
    setWorkNode();
    setIsModalOpen(false);
  };
  const handleCancel = () => {
    setIsModalOpen(false);
  };

  function delay(time) {
    return new Promise((resolve) => setTimeout(resolve, time));
  }

  async function setWorkNode() {
    const response = await api.setNode(checkedList);
    console.log(response);
  }

  async function stopRun() {
    const response = await stopProcess();
    console.log(response);
  }

  async function viewNode() {
    const response = await getNodeDetail();
    var all_detail = [];
    var nodeDetail = { id: "", name: "" };
    for (let i = 0; i < Object.keys(response.data).length; i++) {
      // nodeDetail["id"] = Object.keys(response.data)[i];
      // nodeDetail["name"] = response.data[Object.keys(response.data)[i]].node_name;
      // console.log(nodeDetail);
      all_detail.push(response.data[Object.keys(response.data)[i]].node_name);
      // nodeDetail = { id: "", name: "" };
    }
    setUseNode(all_detail);
    console.log(useNode);
    showModal();
  }

  async function callProcess() {
    // var all_result = {};
    // const response = await fetch('http://127.0.0.1:5000/offline');
    // const myJson = await response.json(); //extract JSON from the http response
    // document.getElementById("demo").innerHTML = "connect: "+myJson.message;
    // setDataTest(myJson.message)
    const body = { input: inputPath, save_fol: outputPath };
    api.createProcessGPS(body);
    // console.log(response)
    // setResApi(response.data.uid);
    // console.log(response.data.uid);
    // for (let i = 0; i < Object.keys(response.data).length; i++) {
    //   // console.log("message: " + response.data[Object.keys(response.data)[i]]);
    //   var node_name = response.data[Object.keys(response.data)[i]].node_name;
    //   result_detail["status"] = (response.data[Object.keys(response.data)[i]].status);
    //   result_detail["all_img"] = Number(
    //     response.data[Object.keys(response.data)[i]].all_img
    //   );
    //   result_detail["process"] = Number(
    //     response.data[Object.keys(response.data)[i]].progress
    //   );
    //   all_result[node_name] = result_detail;
    //   // all_result.push(result_detail);
    //   // all_result[result_detail["id"]] = result_detail;
    //   result_detail = {
    //     id: "",
    //     process: 0,
    //     all_img: 0,
    //     status: "",
    //   };
    // }
    // console.log(response.data);
    // setResAllResult(response.data);
    // console.log(resAllResult);
    // setResProgress();
    await delay(3000);
    // progressWindow();
    // checkProgress();
    const interval = await setInterval(() => {
      console.log("call task");
      fetch("http://localhost:5000/task")
        .then((res) => res.json())
        .then((json) => {
          // console.log(json);
          // all_result = [];
          for (let i = 0; i < Object.keys(json).length; i++) {
            // console.log(json[Object.keys(json)[i]].status)
            if (
              json[Object.keys(json)[i]].status === "complete" ||
              json[Object.keys(json)[i]].status === "stop"
            ) {
              // all_result[i]["id"] = json[Object.keys(json)[i]].node_name;
              // all_result[json[Object.keys(json)[i]].node_name]["status"] =
              //   json[Object.keys(json)[i]].status;
              // all_result[i]["all_img"] = Number(
              //   json[Object.keys(json)[i]].all_img
              // );
              // all_result[json[Object.keys(json)[i]].node_name]["process"] = Number(
              //   json[Object.keys(json)[i]].progress
              // );
              // all_result.push(result_detail);
              // result_detail = {
              //   id: "",
              //   process: 0,
              //   all_img: 0,
              //   status: "",
              // };
              // setResProgress(Number(json[Object.keys(json)[i]].progress));
              // setResStatus(json[Object.keys(json)[i]].status);
              clearInterval(interval);
              // console.log(json[Object.keys(json)[i]].status);
            } else {
              // setResProgress(Number(json[Object.keys(json)[i]].progress));
              // all_result[i]["status"] = json[Object.keys(json)[i]].status;
              // all_result[i]["all_img"] = Number(
              //   json[Object.keys(json)[i]].all_img
              // );
              // all_result[json[Object.keys(json)[i]].node_name]["process"] =
              //   Number(json[Object.keys(json)[i]].progress);
              console.log(
                "check job: " +
                  json[Object.keys(json)[i]].status +
                  " : " +
                  json[Object.keys(json)[i]].process
              );
            }
          }
          // setResFolProgress(json[Object.keys(json)[0]]);
          setResAllResult(json);
          console.log(resAllResult);
        });
    }, 5 * 1000);
  }

  // async function progressWindow(){
  //   // show_progress = []
  //     return resAllResult.map((Result) => (
  //       <div key={Result.id}>
  //         <h3>Complete file: {Result.process}</h3>
  //         <h3>Total file: {Result.all_img}</h3>
  //         <Progress
  //           type="circle"
  //           percent={(Result.process / Result.all_img) * 100}
  //           showInfo={false}
  //           size={50}
  //         />
  //       </div>
  //       // <div key={station.call} className='station'>{station.call}</div>
  //     ));
  // }

  const onFinish = (values) => {
    console.log("Success:", values);
  };
  const onFinishFailed = (errorInfo) => {
    console.log("Failed:", errorInfo);
  };
  return (
    <div>
      <div>
        <Card
          // title="Card title"
          bordered={false}
          // style={{
          //   width: 300,
          // }}
        >
          <Form
            name="basic"
            labelCol={{
              span: 8,
            }}
            wrapperCol={{
              span: 16,
            }}
            style={{
              maxWidth: 600,
            }}
            initialValues={{
              remember: true,
            }}
            onFinish={callProcess}
            onFinishFailed={onFinishFailed}
            autoComplete="off"
          >
            <Form.Item
              label="Input path"
              name="input_path"
              rules={[
                {
                  required: true,
                  message: "Please input your username!",
                },
              ]}
              onChange={(e) => setInputPath(e.target.value)}
            >
              <Input />
            </Form.Item>

            <Form.Item
              label="Output path"
              name="output_path"
              rules={[
                {
                  required: true,
                  message: "Please input your password!",
                },
              ]}
              onChange={(e) => setOutputPath(e.target.value)}
            >
              <Input />
            </Form.Item>

            <Form.Item
              wrapperCol={{
                offset: 8,
                span: 16,
              }}
            >
              <Button type="primary" htmlType="submit">
                Start process
              </Button>
            </Form.Item>
          </Form>
        </Card>
      </div>
      <div style={{ marginTop: "20px" }}>
        <Row>
          <Col span={8}>Inpainting progress</Col>
          <Col span={8} offset={8}>
            <Space size="large">
              <Button
                type="primary"
                style={{ backgroundColor: "Green" }}
                htmlType="submit"
                onClick={viewNode}
              >
                Add node
              </Button>
              <Modal
                title="Select client node"
                open={isModalOpen}
                onOk={handleOk}
                onCancel={handleCancel}
              >
                <Checkbox
                  indeterminate={indeterminate}
                  onChange={onCheckAllChange}
                  checked={checkAll}
                >
                  Check all
                </Checkbox>
                <Divider />
                <CheckboxGroup
                  options={useNode}
                  value={checkedList}
                  onChange={onChange}
                />
                {/* <ul>
                {useNode.map((useNode) => (
                  <li key={useNode.id}>node{useNode.name}</li>
                ))}
              </ul> */}
              </Modal>
              <Button htmlType="submit" type="primary" danger onClick={stopRun}>
                Stop all
              </Button>
            </Space>
          </Col>
        </Row>
      </div>
      {/* <div>
          <Divider orientation="left" plain>
            <Flex justify="flex-start" align="flex-start">
              Inpainting progress
            </Flex>
          </Divider>
          <Flex justify="flex-end" align="flex-start">
            <Space size="large">
              <Button
                type="primary"
                style={{ backgroundColor: "Green" }}
                htmlType="submit"
                onClick={viewNode}
              >
                Add node
              </Button>
              <Modal
                title="Basic Modal"
                open={isModalOpen}
                onOk={handleOk}
                onCancel={handleCancel}
              >
                <Checkbox
                  indeterminate={indeterminate}
                  onChange={onCheckAllChange}
                  checked={checkAll}
                >
                  Check all
                </Checkbox>
                <Divider />
                <CheckboxGroup
                  options={useNode}
                  value={checkedList}
                  onChange={onChange}
                />
              </Modal>
              <Button htmlType="submit" type="primary" danger onClick={stopRun}>
                Stop all
              </Button>
            </Space>
          </Flex>
        </div> */}
      <div style={{ marginTop: "20px" }}>
        <Flex
          // gap="middle"
          // justify="space-evenly"
          // align="center"
          vertical="True"
        >
          {/* {resAllResult[0]} */}
          {/* <h3>Complete file: {resProgress}</h3>
            <h3>Total file: {resAllPic}</h3>
            <Progress
              type="circle"
              percent={(resProgress / resAllPic) * 100}
              showInfo={false}
              size={50}
            /> */}
          {Object.keys(resAllResult).length > 0 && (
            // Folder progress
            <Progress
              percent={Math.floor(
                (resAllResult[Object.keys(resAllResult)[0]].num_folder /
                  resAllResult[Object.keys(resAllResult)[0]].all_folder) *
                  100
              )}
              status={
                resAllResult[Object.keys(resAllResult)[0]].status === "stop" &&
                "exception"
              }
            />
          )}
          {Object.entries(resAllResult).map(([key, value]) => (
            <div key={key}>
              <Card
                title={value.process_folder}
                bordered={false}
                // style={{
                //   width: 300,
                // }}
              >
                <Row>
                  <Col span={6}>
                    <h2>node{value.node_name}</h2>
                  </Col>
                  <Col span={6}>
                    <h3>Complete file: {value.process}</h3>
                  </Col>
                  <Col span={6}>
                    <h3>Total file: {value.all_img}</h3>
                  </Col>
                  <Col span={6}>
                    <Progress
                      type="circle"
                      percent={Math.floor(
                        (value.process / value.all_img) * 100
                      )}
                      // showInfo={false}
                      size={50}
                      status={value.status === "stop" && "exception"}
                    />
                  </Col>
                </Row>
              </Card>
            </div>
          ))}
        </Flex>
      </div>
    </div>
  );
};

// export default Inpaint;
