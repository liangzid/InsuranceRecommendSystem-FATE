/**
 * 小程序配置文件
 */

var host = 'http://127.0.0.1:5000';
var config = {
  // 下面的地址配合云端 Demo 工作
  service: {
    host,
    // 登录地址，用于建立会话
    loginUrl: `${host}/login`,
    // 测试的请求地址，用于测试会话
    requestUrl: `${host}/test`,
    // 下载服务接口
    tunnelUrl: `${host}/download`,
    // 上传表单接口
    uploadUrl: `${host}/upload`,
    //支付接口
    payUrl: `${host}/api/pay`
  }
};
module.exports = config;