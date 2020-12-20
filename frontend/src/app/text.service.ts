import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class TextService {

  constructor(private http: HttpClient) { }

  submitText(text: any) {
    const url = environment.apiUrl + '/rest/process-text/' + encodeURIComponent(text);
    return this.http.get(url);
  }

}
