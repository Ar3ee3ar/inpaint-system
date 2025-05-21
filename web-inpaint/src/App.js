import { Layout, Space } from "antd";
import { BrowserRouter, Routes, Route, Link, Router } from "react-router-dom";

import logo from "./logo.svg";
import "./App.css";
import {Inpaint, BridgeHeader} from "./page/inpaint_process";
import { InpaintNakorn, NakornHeader } from "./page/inpaint_process_nakorn";
import { GPSInpaint, GPSHeader } from "./page/gps_process";
import { Color } from "antd/es/color-picker";
import { CarOutlined, HomeOutlined } from "@ant-design/icons";
const { Header, Content, Footer, Sider } = Layout;

const App = () => {
  return (
    <div className="App">
      <Layout>
        <Header>
          <BrowserRouter>
            <Routes>
              {/* Root of Equation */}
              <Route exact path="/" element={<GPSHeader />} />
              <Route exact path="/bridge" element={<BridgeHeader />} />
              <Route exact path="/nakorn" element={<NakornHeader />} />
            </Routes>
          </BrowserRouter>
        </Header>
        <Layout>
          {/* <Sider>left sidebar</Sider> */}
          <Content style={{ margin: 10 }}>
            <BrowserRouter>
              <Routes>
                {/* Root of Equation */}
                <Route exact path="/" element={<GPSInpaint />} />
                <Route exact path="/bridge" element={<Inpaint />} />
                <Route exact path="/nakorn" element={<InpaintNakorn />} />
              </Routes>
            </BrowserRouter>
            {/* <Inpaint /> */}
          </Content>
          {/* <Sider>right sidebar</Sider> */}
        </Layout>
        {/* <Footer>footer</Footer> */}
      </Layout>
    </div>
  );
};

export default App;
