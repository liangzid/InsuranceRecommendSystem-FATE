//app.js
App({
  onLaunch: function () {
    if (!swan.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力');
    } else {
      swan.cloud.init({
        traceUser: true
      });
    }

    this.globalData = {};
  },
  onLaunch: function () {
    console.log("start miniapp");
  }
});