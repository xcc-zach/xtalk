export { Platform, getPlatform }

declare const uni: any;

enum Platform {
    MpWeixin,
    Web
}

function getPlatform(): Platform {
    if (typeof window !== "undefined" && typeof document !== "undefined")
      return Platform.Web;
    const hasUniRuntime = typeof uni !== "undefined" && typeof uni.getSystemInfoSync === "function";
    if (hasUniRuntime){
      return Platform.MpWeixin;
    }
    throw new Error("getPlatform: Unknown platform");
}