const pageUrl = window.location.href;

//const params = new URLSearchParams(window.location.search);
//const userNameObj = document.getElementById('currentProfile');
const userNameParam = 'antonio.marinelli@ibm.com';
//const userFullNameParam = userNameObj.options[userNameObj.selectedIndex].text;
const userFullNameParam = "Antonio"

// Funzione per generare un UUID
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0,
            v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function getBrowserName() {
    const ua = navigator.userAgent;

    if (ua.includes("Firefox/")) return "Firefox";
    if (ua.includes("Edg/")) return "Edge";
    if (ua.includes("Chrome/") && !ua.includes("Edg/")) return "Chrome";
    if (ua.includes("Safari/") && !ua.includes("Chrome/")) return "Safari";
    if (ua.includes("OPR/") || ua.includes("Opera/")) return "Opera";

    return "Unknown";
}

function getOS() {
    const platform = navigator.platform.toLowerCase();
    const ua = navigator.userAgent.toLowerCase();

    if (platform.includes("win")) return "Windows";
    if (platform.includes("mac")) return "Mac OS";
    if (platform.includes("linux")) return "Linux";
    if (/android/.test(ua)) return "Android";
    if (/iphone|ipad|ipod/.test(ua)) return "iOS";

    return "Unknown";
}

function getUserName() {
    return userNameParam;
}

function getOrCreateSessionId() {
    const key = 'custom_session_id';
    const timestampKey = 'session_timestamp_';
    const sessionTimeout = 30 * 60 * 1000; // 30 minuti

    // sessionStorage = unico per ogni tab
    let id = sessionStorage.getItem(key);

    const now = Date.now();

    if (!id) {
        // Nuovo tab = nuova sessione
        id = generateUUID();
        sessionStorage.setItem(key, id);
        localStorage.setItem(timestampKey + id, now.toString());
        console.log('🆕 Nuova sessione per questo tab:', id);
    } else {
        // Controlla se la sessione è scaduta
        let lastActivity = localStorage.getItem(timestampKey + id);

        if (!lastActivity || (now - parseInt(lastActivity)) > sessionTimeout) {
            // Sessione scaduta, creane una nuova
            id = generateUUID();
            sessionStorage.setItem(key, id);
            localStorage.setItem(timestampKey + id, now.toString());
            console.log('🆕 Sessione scaduta, nuova per questo tab:', id);
        } else {
            // Estendi la sessione esistente
            localStorage.setItem(timestampKey + id, now.toString());
            console.log('♻️ Sessione tab esistente estesa:', id);
        }
    }

    return id;
}


function getOrCreateBrowser() {
    const browser = 'browser';
    let browser_id = sessionStorage.getItem(browser);
    if (!browser_id) {
        browser_id = getBrowserName();
        sessionStorage.setItem(browser, browser_id);
    }
    return browser_id;
}

function getOrCreateOS() {
    const os = 'os';
    let os_id = sessionStorage.getItem(os);
    if (!os_id) {
        os_id = getOS();
        sessionStorage.setItem(os, os_id);
    }
    return os_id;
}

function getOrCreateUserName() {
    const userName = 'userName';
    let userName_id = sessionStorage.getItem(userName);
    if (!userName_id) {
        userName_id = getUserName();
        sessionStorage.setItem(userName, userName_id);
    }
    return userName_id;
}

const customSessionId = getOrCreateSessionId();
const browser = getOrCreateBrowser();
const os = getOrCreateOS();
const userName = getOrCreateUserName();

function preSendHandler(event) {
    //const triggerAgent = document.getElementById("aiAgentEnabler").checked;

    // This will set the assistant (session) variable "User_Name" to the name of our user. In your deployed environment,
    // you could retrieve this name from some sort of user profile object that is available in the application. In this
    // tutorial, we are just hard-coding the username to "Cade". The code below also ensures that if there are already
    // other context values in the message, that we will leave them alone. Also, this code only runs for the initial
    // welcome message, but you could set the variable on any message that is sent to the assistant.

    //if (event.data.history) {
    // Make sure these objects exist but don't override them if they already do.
    // Note: If you are using a dialog skill instead of an actions skill, then replace "actions skill" with
    // "main skill" and replace "skill_variables" with "user_defined".
    event.data.context.skills['actions skill'] = event.data.context.skills['actions skill'] || {};
    event.data.context.skills['actions skill'].skill_variables = event.data.context.skills['actions skill'].skill_variables || {};
    event.data.context.skills['actions skill'].skill_variables.custom_session_id = customSessionId;
    //event.data.context.skills['actions skill'].skill_variables.browser = browser;
    //event.data.context.skills['actions skill'].skill_variables.os = os;
    //event.data.context.skills['actions skill'].skill_variables.userName = userName;
    //event.data.context.skills['actions skill'].skill_variables.newProfile = newProfile;
    //event.data.context.skills['actions skill'].skill_variables.targetProfile = targetProfile;
    //event.data.context.skills['actions skill'].skill_variables.triggerAgent = triggerAgent;
}

// Funzione che viene eseguita al caricamento dell'istanza
async function onLoad(instance) {
    instance.updateLauncherGreetingMessage("Ciao. Fammi sapere se posso darti una mano!");

    instance.on({ type: 'pre:send', handler: preSendHandler });

    let initialMessage = '';
    let lab1 = '', lab2 = '', lab3 = '';
    let lab4 = '', lab5 = '', lab6 = '';
    let lab7 = '', lab8 = '', lab9 = '', lab10 = '', lab11 = '';

    if (pageUrl.toLowerCase().includes("/")) {
        initialMessage = "Ciao, come posso aiutarti?";

        lab1 = "L'anno scorso ho avuto un posto alloggio. Posso averlo anche quest'anno?";
        lab2 = "Mi hanno comunicato che sono beneficiario del posto alloggio. Ma ora cosa devo fare?";
        lab3 = "Come viene calcolato il punteggio di merito per un laureando magistrale che ha anche una carriera triennale precedente?";
        lab4 = "I've been informed that I've been awarded accommodation. What should I do now?";
        lab5 = "L'an dernier, j'ai eu un logement. Puis-je l'obtenir aussi cette année ?";
        lab6 = "On m'a informé que j'ai obtenu un logement. Que dois-je faire maintenant ?";
        lab7 = "El año pasado tuve alojamiento. ¿Puedo obtenerlo también este año?";
        lab8 = "Me han informado de que me han concedido alojamiento. ¿Qué debo hacer ahora?";
        lab9 = "Letztes Jahr hatte ich eine Unterkunft. Kann ich sie auch dieses Jahr bekommen?";
        lab10 = "Mir wurde mitgeteilt, dass ich eine Unterkunft erhalten habe. Was soll ich jetzt tun?";
        lab11 = "通話を録音した後にエラーに気付いた場合、メリットスコアを変更するにはどのような手順を踏む必要がありますか?"
    }

    instance.updateHomeScreenConfig({
        is_on: true,
        greeting: initialMessage,
        starters: {
            is_on: true,
            buttons: [
                { label: lab1 }, { label: lab2 }, { label: lab3 },
                { label: lab4 }, { label: lab5 }, { label: lab6 },
                { label: lab7 }, { label: lab8 }, { label: lab9 },
                { label: lab10 }, { label: lab11 }
            ]
        }
    });

    await instance.render();

    if (pageUrl.includes("registrazione")) {
        setTimeout(() => {
            instance.openWindow();
        }, 5000);
    }
}


// Configurazione dell'istanza di Watson Assistant
window.watsonAssistantChatOptions = {
    integrationID: "ee5b8d96-66e4-44d0-b64a-b9b3a7dfaed0", // The ID of this integration.
    region: "wxo-us-south", // The region your integration is hosted in.
    serviceInstanceID: "8a4196f0-04c4-4878-bba0-e0d22a03c401",
    disableSessionHistory: false,
    orchestrateUIAgentExtensions: false, //If you wish to enable optional UI Agent extensions.
    disableSessionHistory: true,
    onLoad: onLoad,
    disclaimer: {
      isOn: false,
      disclaimerHTML: `
        <div style="font-size: 14px; line-height: 1.5;">
          <strong>Nota importante:</strong> le risposte di questo assistente possono essere generate da sistemi di intelligenza artificiale.
          Non inserire dati personali o sensibili. <a href="https://www.ibm.com/privacy" target="_blank">Informativa sulla privacy</a>.
        </div>
      `
    }
};


setTimeout(function () {
    const t = document.createElement('script');
    t.src = "https://web-chat.global.assistant.watson.appdomain.cloud/versions/" +
        (window.watsonAssistantChatOptions.clientVersion || 'latest') +
        "/WatsonAssistantChatEntry.js";
    document.head.appendChild(t);
});
