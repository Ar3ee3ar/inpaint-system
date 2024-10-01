import { Layout, Space } from "antd";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

import logo from "./logo.svg";
import "./App.css";
import {Inpaint} from "./page/inpaint_process";
import { Color } from "antd/es/color-picker";
import { CarOutlined, HomeOutlined } from "@ant-design/icons";
const { Header, Content, Footer, Sider } = Layout;

const App = () => {
  return (
    <div className="App">
      <Layout>
        <Header>
          <h1 style={{ color: "white", textAlign: "left", margin: 0 }}>
            <Space>
              <CarOutlined />
              Remove car
            </Space>
          </h1>
        </Header>
        <Layout>
          {/* <Sider>left sidebar</Sider> */}
          <Content style={{ margin: 10 }}>
            <Inpaint />
          </Content>
          {/* <Sider>right sidebar</Sider> */}
        </Layout>
        {/* <Footer>footer</Footer> */}
      </Layout>
    </div>
  );
};

export default App;
